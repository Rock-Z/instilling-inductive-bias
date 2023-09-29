import sys
import os

# Currently NeuroSurgeon is not a package, add path manually to import
sys.path.append("../")
from NeuroSurgeon.src.Models.circuit_model import CircuitConfig, CircuitModel
from NeuroSurgeon.src.Masking.contsparse_layer import ContSparseLayer

import transformers
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D # This is the GPT "Linear" layer

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from utils import generate_samples, format, set_random_seed, Numbers

from tqdm import tqdm
import yaml
import shutil

import wandb
# Controls whether WandB logging is enabled
WANDB = True

def compute_metrics(eval_pred, circuit_model):
    """Given evaluation prediction tuple, compute metris and return as dict

    Args:
        eval_pred (tuple(Tensor, Tensor)): tuple of prediction Tensor of shape (batch_size, seq_len, vocab_size) and 
        label of shape (batch_size, seq_len)

    Returns:
        dict: Dictionary containing prediction accuracy (computed with label of last token only), along with 
        ratio of nonzero params and mean of mask weights
    """
    predictions, labels = eval_pred
    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    predictions = np.argmax(predictions[:, -2, :], axis=1)
    correct = np.sum([predictions[i] == labels[i, -1] for i in range(len(predictions))])
    
    # Calculate percentage of masked params
    nonzero_params = 0
    total_params = 0
    mean = 0
    n_modules = 0
    for module in circuit_model.modules():
        if isinstance(module, ContSparseLayer):
            n_modules += 1
            mean += float(torch.mean(module.weight_mask_params))
            mask = module._compute_mask("weight_mask_params")
            nonzero_params += int(torch.sum(mask > 0))
            total_params += torch.numel(mask)
    
    
    return {
        "accuracy" : correct/len(predictions),     
        # Strictly speaking this is not exactly "mean", as # of params in each layer is different
        "mask_param_mean": round(mean / n_modules, 4), 
        "nonzero_params_percentage": round(nonzero_params / total_params, 4),
    }

def load_model(train_configs : dict, circuit_configs : dict):
    """Load instance of `CircuitModel` based on given configuration dictionaries

    Args:
        train_configs (dict): `train_configs` as expected in the yaml config file
        circuit_configs (dict): `circuit_configs` as expected in the yaml config file

    Returns:
        CircuitModel: Masked model initialized from `GPT2LMHeadModel`
    """

    model = GPT2LMHeadModel.from_pretrained(train_configs["model_path"])

    # Since GPT uses Conv1D for all MLPs & Attn layers, only Conv1D layers need to be masked
    layers_to_ablate = []

    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            layers_to_ablate.append(name)

    print(format("Layers passed into CircuitModel:\n", "BOLD"), layers_to_ablate)

    # Initialize configs and masking procedure
    circuit_config = CircuitConfig(
        target_layers=layers_to_ablate,
        **circuit_configs
    )

    # Define the masked model on which the subnetwork is trained
    circuit_model = CircuitModel(circuit_config, model)
    
    return circuit_model


def main():
    # Load configuration
    assert len(sys.argv) == 2, "Usage: ablate.py <config.yaml>"

    with open(sys.argv[1], 'r') as file:
        args = yaml.safe_load(file)
        reproducibility_configs = args["reproducibility_configs"]
        data_configs = args["data_configs"]
        dataloader_configs = args["dataloader_configs"]
        circuit_configs = args["circuit_configs"]
        train_configs = args["train_configs"]
        optim_configs = args["optim_configs"]

    # Copy config file to output path
    os.makedirs(train_configs['output_path'], exist_ok=True)
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

    # Load debug status
    verbose = train_configs["verbose"]

    # Load model
    circuit_model = load_model(train_configs, circuit_configs)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == "cpu":
        Warning("CUDA not available, training on CPU")
    circuit_model.to(device)    

    # Generate samples
    X = generate_samples(**data_configs)
    # Eval set always has n_samples == max
    data_configs["n_samples"] = data_configs["max"]
    X_eval = generate_samples(**data_configs) 

    ablation_data, eval_data = Numbers(X), Numbers(X_eval)
    dataloader = DataLoader(ablation_data, **dataloader_configs)

    optimizer = getattr(optim, optim_configs["optim_name"])\
        (circuit_model.parameters(), **{i:optim_configs[i] for i in optim_configs if i != "optim_name"})

    if verbose:
        print(format("Initialized optimizer:\n", "BOLD"), optimizer)

    if WANDB:
        # WandB logging
        wandb.login()

        run = wandb.init(
            project= "numbers-ablation",
            config = args
        )

    best_acc = 0
    temp = 1
    for epoch in range(train_configs["n_epochs"]):
        circuit_model.train(True)
        
        running_loss = 0.
        last_loss = 0.
        
        with tqdm(total = len(dataloader)) as pbar:
            pbar.set_description(f"epoch {epoch + 1} batch 0 loss: {last_loss}")
            pbar.set_postfix({"temp": round(temp, 2)})
            for i, data in enumerate(dataloader):
                outputs = circuit_model(**{i: data[i].to(device) for i in data})
                loss = outputs.loss
                loss.backward()
                
                # Train!
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
                
                running_loss += outputs.loss.item()
                # Update data 10 times per epoch
                if i % (len(dataloader) // 10) == (len(dataloader) // 10 - 1):
                    last_loss = running_loss / (len(dataloader) // 10)
                    pbar.set_description(f"Epoch {epoch + 1} batch {i + 1} loss: {round(last_loss, 4)}")
                    running_loss = 0.
            
        # Evaluate at epoch end
        circuit_model.eval()    
        with torch.no_grad():        
            labels = torch.from_numpy(X_eval).to(device)
            outputs = circuit_model(input_ids = torch.from_numpy(X_eval).to(device), labels = labels)
            eval_loss = outputs.loss
            predictions = outputs.logits
            
            eval_stats = compute_metrics((predictions, labels), circuit_model)
            eval_stats["eval_loss"] = round(float(eval_loss), 4)
            eval_acc = eval_stats["accuracy"]
        
        # Print eval stats to terminal
        print(format(f"Eval stats at end of epoch {epoch + 1}: {eval_stats}", "YELLOW"))
        
        # Send log data to wandb
        if WANDB:
            eval_stats["temp"] = temp
            eval_stats["last_train_loss"] = last_loss
            wandb.log(eval_stats)
        
        # Increase temperature for each masked layer at epoch end
        # Basically a manual exponential scheduler
        for module in circuit_model.modules():
            if hasattr(module, "temperature"):
                module.temperature *= train_configs["final_temp"] ** (1 / train_configs["n_epochs"])
                temp = module.temperature
        # print(format(f"Temp increased to {temp}", "BLUE"))
                
        # Save model
        torch.save(circuit_model.state_dict(), train_configs["output_path"] + f"/latest_circuit_model.pt")
        if eval_acc >= best_acc:
            best_acc = eval_acc
            print(format(f"Saving model with eval acc {eval_acc}", "BLUE"))
            torch.save(circuit_model.state_dict(), train_configs["output_path"] + f"/best_circuit_model.pt")

if __name__ == "__main__":
    main()