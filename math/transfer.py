import sys
import os

sys.path.append("../")
from NeuroSurgeon.src.Models.circuit_model import CircuitConfig, CircuitModel
from NeuroSurgeon.src.Masking.contsparse_layer import ContSparseLayer

import transformers
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.pytorch_utils import Conv1D

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import pandas as pd

from utils import generate_samples, format, set_random_seed, Numbers

from tqdm import tqdm
import yaml
import shutil

import wandb

def load_model(model_paths: dict, 
               circuit_configs: dict, 
               inherit_from : str, 
               freeze_subnet : bool = True
               ) -> CircuitModel:
    """Given parsed config dictionaries, load and return a CircuitModel ready for subnet transfer training"""
    
    # Must load either the subnetwork (subnet), a newly initialized model (random), or the full model on which the subnet is trained
    assert inherit_from in ["subnet", "subnet_sampled", "subnet_sampled_complement",  "random", "original"], "Invalid inherit_from option"
    
    if inherit_from in ["subnet", "subnet_sampled", "subnet_sampled_complement"]:        
        
        original_model = GPT2LMHeadModel.from_pretrained(model_paths["original"])

        # Recreate the CircuitModel
        layers_to_ablate = []
        for name, module in original_model.named_modules():
            if isinstance(module, Conv1D):
                layers_to_ablate.append(name)

        circuit_config = CircuitConfig(
                target_layers=layers_to_ablate,
                **circuit_configs
            )

        transfer_model = CircuitModel(circuit_config, original_model)

        subnet_state_dict = torch.load(model_paths["subnet"], map_location=original_model.device)
        
        # Make sure the model & subnet ablation state dict are indeed the same model
        for name, param in original_model.named_parameters():
            # Only mask params should have been trained during the ablation process
            if "mask" not in name:
                assert torch.equal(param, subnet_state_dict["root_model." + name]),\
                    f"Param {name} recorded in subnetwork not equal to those in original model"
        
        # Note: here the state_dict is loaded before initializing transfer because our subnet state dict doesn't have
        # the parameters that only exists in a transfer model
        transfer_model.load_state_dict(subnet_state_dict)        
        print(format("Loaded Subnetwork from state dict", "BOLD"))
                
        # Replace mask params with a sampled mask if necessary
        if inherit_from == "subnet_sampled":
            for module in transfer_model.modules():
                if isinstance(module, ContSparseLayer):
                    weight_mask_sampled = module._sample_mask_randomly("weight_mask_params")
                    module.weight_mask_params.data = weight_mask_sampled * 2 - 1
                    if module.mask_bias:
                        bias_mask_sampled = module._sample_mask_randomly("bias_mask_params")
                        module.bias_mask_params.data = bias_mask_sampled * 2 - 1
        elif inherit_from == "subnet_sampled_complement":
            for module in transfer_model.modules():
                if isinstance(module, ContSparseLayer):
                    weight_mask_sampled = module._sample_mask_from_complement("weight_mask_params")
                    module.weight_mask_params.data = weight_mask_sampled * 2 - 1
                    if module.mask_bias:
                        bias_mask_sampled = module._sample_mask_from_complement("bias_mask_params")
                        module.bias_mask_params.data = bias_mask_sampled * 2 - 1            
                        
        transfer_model.init_subnet_transfer()
        transfer_model.train()          
        
        if not freeze_subnet:
            for module in transfer_model.modules():
                if isinstance(module, ContSparseLayer):
                    module.weight_subnet.requires_grad = True
                    if module.mask_bias:
                        module.bias_subnet.requires_grad = True
        
        # FIXME: temporary! this is for testing whether changing initialization scheme has a significant effect
        #with torch.no_grad():
        #    for module in transfer_model.modules():
        #        if isinstance(module, ContSparseLayer):
        #            module.weight.data /= 10
        
        # Sanity check for expected bahavior of params in training mode
        for module in transfer_model.modules():
            if isinstance(module, ContSparseLayer):
                assert hasattr(module, "weight_subnet") and (hasattr(module, "bias_subnet") if module.mask_bias else True), "Subnets not initialized"
                assert module.weight.requires_grad and module.bias.requires_grad, "Params should still be trainable"
                assert ((not module.weight_subnet.requires_grad) == freeze_subnet) \
                    and ((not module.bias_subnet.requires_grad == freeze_subnet) if module.mask_bias else True), "Subnets should be frozen"
                
                mask = module._compute_mask("weight_mask_params")
                num_binary_values = torch.numel(mask[mask == 0]) + torch.numel(mask[mask == 1])
                mask_size = torch.numel(mask)
                assert num_binary_values == mask_size, \
                f"Mask unexpected behavior: {num_binary_values} mask values are binary but mask is of size {mask_size} in module {module}"

    elif inherit_from == "original":
        # Just skip all the transfer code and use the orignal model
        print(format("Transferring the entire model", "BOLD"))
        original_model = GPT2LMHeadModel.from_pretrained(model_paths["original"])
        transfer_model = original_model
            
    elif inherit_from == "random":
        # Only load from configs, not pretrained weights
        print(format("Training from random initialization", "BOLD"))
        config = GPT2Config.from_json_file(model_paths["original"] + "/config.json")
        transfer_model = GPT2LMHeadModel(config)
        
    return transfer_model
    

def main():
    
    # Load configuration
    assert len(sys.argv) == 2, "Usage: transfer.py <config.yaml>"

    with open(sys.argv[1], 'r') as file:
        args = yaml.safe_load(file)
        reproducibility_configs = args["reproducibility_configs"]
        model_paths = args["model_paths"]
        data_configs = args["data_configs"]
        circuit_configs = args["circuit_configs"]
        train_configs = args["train_configs"]
        optim_configs = args["optim_configs"]
    
    # Copy config file to output path
    original_output_path = train_configs["output_path"]
    os.makedirs(original_output_path, exist_ok=True)
    shutil.copyfile(sys.argv[1], train_configs["output_path"] + "/configs.yaml")
    
    # If reproducibility configs on, set random seed to the one provided. Otherwise take one 
    # from systems entropy source and record it
    if reproducibility_configs['reproduce']:
        set_random_seed(reproducibility_configs['random_seed'])
        print(format(f"Reproducing run with random seed {reproducibility_configs['random_seed']}", "BLUE"))
    else:
        seed = int.from_bytes(os.urandom(4), sys.byteorder)
        set_random_seed(seed)        
        print(format(f"Generated & using random seed {seed}", "BLUE"))
        # Record seed in output path
        with open(train_configs["output_path"] + "/seed.txt", "w") as f:
            f.write(str(seed))        
    
    # Controls WandB logging
    WANDB = train_configs["wandb"]
    
    # More terminal output & san checks if True
    verbose = train_configs["verbose"]
    
    accuracy_logs = []
    
    for n_disambiguation in data_configs["n_disambiguation"]:          
        # Generate new data
        # Note: a fair comparison requires different initializations to be trained on the same data
        X_ambiguous = generate_samples(data_configs["max"], data_configs["max"], data_configs["p"], ambiguous=True, computation="a^2 + ab")

        # Inject some number of disambiguation data intro training samples
        X_disambiguation = generate_samples(max([data_configs["max"], n_disambiguation]), 
                                            data_configs["max"], 
                                            data_configs["p"], 
                                            ambiguous=False, 
                                            computation="a^2 + ab")
        
        X_train = np.concatenate((X_ambiguous, X_disambiguation[np.random.choice(len(X_disambiguation), n_disambiguation, replace=False)]))

        train_dataset = Numbers(X_train)
        train_dataloader = DataLoader(train_dataset, batch_size=train_configs["batch_size"], shuffle=True)

        # Unambiguous data used for eval
        X_eval = generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=True, computation="a^2 + ab")
        X_eval_ab= generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=False, computation= "a^2 + ab")
        X_eval_a= generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=False, computation= "2a^2")
        X_eval_b = generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=False, computation= "2b^2")

        eval_datasets = {"original": X_eval, "original_unambiguous": X_eval_ab, "2a^2": X_eval_a, "2b^2": X_eval_b}        
                      
        for inherit_from_option in train_configs["inherit_from"]:
            # Each iteration stored in different folder
            train_configs["output_path"] = original_output_path + f"/{inherit_from_option}/{n_disambiguation}_disambiguation"            
            os.makedirs(train_configs["output_path"], exist_ok=True)      
            
            best_acc = 0
            # Iterate through all given learning rates
            for lr in optim_configs["lr"]:
                # Load model
                transfer_model = load_model(model_paths, circuit_configs, inherit_from_option, train_configs["freeze_subnet"])
                    
                # Put model on a GPU if possible
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                transfer_model.to(device)   

                optimizer = getattr(optim, optim_configs["optim_name"])\
                    (transfer_model.parameters(), lr=lr) 

                if verbose:
                    print(format("Initialized optimizer:\n", "BOLD"), optimizer)

                if WANDB:
                    # WandB logging
                    wandb.login()

                    run = wandb.init(
                        project="numbers_transfer",
                        config=train_configs
                    )
                    
                eval_stats_logs = []
                
                
                print(f"Hyperparams for the run: {inherit_from_option=}, {n_disambiguation=}, {lr=}, {best_acc=}")
                # Initialize progress bar & setup placeholder description
                with tqdm(total = len(train_dataloader) * train_configs["n_epochs"]) as pbar:
                    pbar.set_description(f"epoch 0 batch 0 loss: 0.0000")

                    # Train!
                    for epoch in range(train_configs["n_epochs"]):
                        transfer_model.train(True)
                        
                        running_loss = 0.
                        last_loss = 0.

                        for i, data in enumerate(train_dataloader):
                            outputs = transfer_model(**{i: data[i].to(device) for i in data})
                            loss = outputs.loss
                            loss.backward()
                            
                            # Train!
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                            
                            running_loss += outputs.loss.item()
                            if i % (len(train_dataloader) // 10) == (len(train_dataloader) // 10 - 1):
                                last_loss = running_loss / (len(train_dataloader) // 10)
                                pbar.set_description(f"Epoch {epoch + 1} batch {i + 1} loss: {round(last_loss, 4)}")
                                running_loss = 0.
                            
                        # Evaluate at epoch end
                        transfer_model.eval()    
                        with torch.no_grad():      
                            eval_stats = {}  
                            for dataset_name in eval_datasets:
                                eval_data = eval_datasets[dataset_name]
                                labels = torch.from_numpy(eval_data)
                                outputs = transfer_model(input_ids = torch.from_numpy(eval_data).to(device), 
                                                        labels = labels.to(device))
                                
                                eval_loss = outputs.loss
                                predictions = outputs.logits.cpu().numpy()
                                
                                predictions = np.argmax(predictions[:, -2, :], axis=1)
                                correct = np.sum([predictions[i] == labels[i, -1] for i in range(len(predictions))])  
                                eval_acc = correct/len(predictions)  
                                
                                # Add dataset name before key of eval_stats dict
                                eval_stats[dataset_name + "_eval_acc"] = round(float(eval_acc), 4)                
                                eval_stats[dataset_name + "_eval_loss"] = round(float(eval_loss), 4)
                                
                            eval_acc = eval_stats["original_unambiguous_eval_acc"]
                        
                        eval_stats_logs.append(eval_stats)
                        
                        # Print eval stats to terminal
                        if verbose:
                            print(format(f"Eval stats at end of epoch {epoch + 1}: {eval_stats}", "YELLOW"))
                        
                        # Send log data to wandb
                        if WANDB:
                            eval_stats["last_train_loss"] = last_loss
                            wandb.log(eval_stats)
                                
                        # Save model
                        torch.save(transfer_model.state_dict(), train_configs["output_path"] + f"/latest_transfer_model.pt")
                        if eval_acc >= best_acc:
                            best_acc = eval_acc
                            torch.save(transfer_model.state_dict(), train_configs["output_path"] + f"/best_transfer_model.pt")                            
                            eval_stats_logs_df = pd.DataFrame.from_records(eval_stats_logs)
                            eval_stats_logs_df.to_csv(train_configs["output_path"] + "/eval_stats_logs.csv")
                            print(format(f"Saved model with eval acc {eval_acc}", "BLUE"))                        
                            print('\033[1A\033[1A\033[1A')
                
                # Stop if reached perfect evaluation accuracy with one learning rate, since there are
                # no additional evaluation metrics to help us decide on different learning rates anyway
                if best_acc == 1:
                    break
                        
            # Re-load best model
            best_state_dict = torch.load(train_configs["output_path"] + f"/best_transfer_model.pt")
            transfer_model.load_state_dict(best_state_dict)
                
            # Evaluate on independent test set
            X_test= generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=False, computation= "a^2 + ab")
            
            with torch.no_grad():
                predictions = transfer_model(input_ids = torch.from_numpy(X_test).to(device)).logits.cpu().numpy()
                predictions = np.argmax(predictions[:, -2, :], axis=1)
                best_test_acc = np.sum([predictions[i] == X_test[i, -1] for i in range(len(predictions))]) / len(predictions)
                
                print(format(f"Test accuracy trained with {n_disambiguation} disambiguation data and {inherit_from_option=}: {best_test_acc}", "BOLD"))
                
            accuracy_logs.append({"n_disambiguation": n_disambiguation, 
                                  "inherit_from": inherit_from_option,
                                  "best_eval_acc": best_acc, 
                                  "best_test_acc": best_test_acc,
                                  "lr": lr})
    
    accuracy_logs = pd.DataFrame.from_records(accuracy_logs)
    accuracy_logs.to_csv(original_output_path + "/accuracy_logs.csv")

if __name__ == "__main__":    
    main()