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
from transfer import load_model

from tqdm import tqdm
import yaml
import shutil

import wandb    

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
        X_train = generate_samples(max([data_configs["max"], n_disambiguation]), data_configs["max"], data_configs["p"], computation=data_configs["computation"])

        if n_disambiguation < data_configs["max"]:
            X_train = X_train[:n_disambiguation]

        train_dataset = Numbers(X_train)
        train_dataloader = DataLoader(train_dataset, batch_size=train_configs["batch_size"], shuffle=True)

        # Unambiguous data used for eval
        X_eval = generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=True, computation=data_configs["computation"])
        eval_datasets = {data_configs["computation"]: X_eval}     
                      
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
                            if i % max((len(train_dataloader) // 10), 1) == (len(train_dataloader) // 10 - 1):
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
                                eval_stats["eval_acc"] = round(float(eval_acc), 4)                
                                eval_stats["eval_loss"] = round(float(eval_loss), 4)
                            
                            eval_acc = eval_stats["eval_acc"]
                        
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
            X_test= generate_samples(1000, data_configs["max"], data_configs["p"], ambiguous=False, computation= data_configs["computation"])
            
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