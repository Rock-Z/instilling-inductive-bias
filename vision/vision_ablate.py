import sys
import os

# Currently NeuroSurgeon is not a package, add path manually to import
sys.path.append("../")
from NeuroSurgeon.src.Models.circuit_model import CircuitConfig, CircuitModel
from NeuroSurgeon.src.Masking.contsparse_layer import ContSparseLayer

import numpy as np
from tqdm import tqdm
import shutil
import copy
import yaml
import wandb

import torch
from torch import optim
from torch.nn import Conv2d, Linear
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ResNetModel, ResNetConfig
from transformers import AutoModelForImageClassification
from transformers import AutoFeatureExtractor, ResNetForImageClassification, ViTForImageClassification
from transformers import Trainer, TrainingArguments

from utils import ImageNet16Class, format

WANDB = True

def set_random_seed(seed : int):
    """Fix torch and numpy seed for run reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)    
    
def split_train_eval(imnet : ImageNet16Class, test_ratio : float = 0.1):
    """Split ImageNet16Class dataset into train and test sets"""
    eval_set = copy.deepcopy(imnet)
    train_set = copy.deepcopy(imnet)
    for i, image_class in enumerate(imnet.image_names):
        eval_set.image_names[i] = image_class[:int(len(image_class) * test_ratio)]
        train_set.image_names[i] = image_class[int(len(image_class) * test_ratio):]
        
    eval_set.build_dataset()
    train_set.build_dataset()
    
    return train_set, eval_set

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def load_model(train_configs, circuit_configs):
    # Load pretrained ResNet previous adapted to ImageNet16Class
    model = AutoModelForImageClassification.from_pretrained(train_configs["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create CircuitModel for masking
    if isinstance(model, ViTForImageClassification):
        layers_to_ablate = list(map(lambda i: i[0], filter(lambda i: isinstance(i[1], Conv2d) or isinstance(i[1], Linear), model.named_modules())))
    elif isinstance(model, ResNetForImageClassification):
        layers_to_ablate = list(map(lambda i: i[0],filter(lambda i: isinstance(i[1], Conv2d), model.named_modules())))
    else: 
        # Raise error if model is not supported
        raise NotImplementedError(f"Model {type(model)} not supported")

    circuit_config = CircuitConfig(
        target_layers=layers_to_ablate,
        **circuit_configs
        )    
    circuit_model = CircuitModel(circuit_config, model)
    circuit_model.to(device)
    
    return circuit_model, device

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

    predictions = np.argmax(predictions, axis=1)
    correct = np.sum(predictions == labels)
    
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

def main():
    # Load configuration
    assert len(sys.argv) == 2, "Usage: ablate.py <config.yaml>"

    with open(sys.argv[1], 'r') as file:
        args = yaml.safe_load(file)
        reproducibility_configs = args["reproducibility_configs"]
        dataloader_configs = args["dataloader_configs"]
        circuit_configs = args["circuit_configs"]
        train_configs = args["train_configs"]
        optim_configs = args["optim_configs"]
        
    hf_pretrained = train_configs["hf_pretrained"]

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

    # No longer needed because dataset is pre-split and stored
    #imnet_16class_pooled = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/")
    #train_set, eval_set = split_train_eval(imnet_16class_pooled, test_ratio=0.1)
    #assert len(train_set) + len(eval_set) == len(imnet_16class_pooled), "Splitting got unexpected results"
    
    train_set = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/train")
    eval_set = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    
    # Load feature extractor into dataset
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_pretrained)
    train_set.transform = lambda input: feature_extractor(input, return_tensors="pt")["pixel_values"]
    eval_set.transform = lambda input: feature_extractor(input, return_tensors="pt")["pixel_values"]
    
    train_dataloader = DataLoader(train_set, collate_fn=collate_fn, **dataloader_configs)
    eval_dataloader = DataLoader(eval_set, collate_fn=collate_fn,**dataloader_configs)

    circuit_model, device = load_model(train_configs, circuit_configs)
    
    optimizer = getattr(optim, optim_configs["optim_name"])\
        (circuit_model.parameters(), **{i:optim_configs[i] for i in optim_configs if i != "optim_name"})
        
    if verbose:
        print(format("Initialized optimizer:\n", "BOLD"), optimizer)

    if WANDB:
        # WandB logging
        wandb.login()

        run = wandb.init(
            project= train_configs["wandb_project"] if "wandb_project" in train_configs else "subnet_discovery",
            config = args
        )

    best_acc = 0
    temp = 1
    for epoch in range(train_configs["n_epochs"]):
        circuit_model.train(True)
        
        running_loss = 0.
        last_loss = 0.
        
        with tqdm(total = len(train_dataloader)) as pbar:
            pbar.set_description(f"epoch {epoch + 1} batch 0 loss: {last_loss}")
            pbar.set_postfix({"temp": round(temp, 2)})
            for i, data in enumerate(train_dataloader):
                outputs = circuit_model(**{i: data[i].to(device) for i in data})
                loss = outputs.loss
                loss.backward()
                
                # Train!
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
                
                running_loss += outputs.loss.item()
                # Update data 10 times per epoch
                if i % (len(train_dataloader) // 10) == (len(train_dataloader) // 10 - 1):
                    last_loss = running_loss / (len(train_dataloader) // 10)
                    pbar.set_description(f"Epoch {epoch + 1} batch {i + 1} loss: {round(last_loss, 4)}")
                    running_loss = 0.
            
        # Evaluate at epoch end
        circuit_model.eval()    
        with torch.no_grad():    
            predictions = []
            labels = []
            eval_losses = []
            
            for i, data in enumerate(eval_dataloader):    
                outputs = circuit_model(**{i: data[i].to(device) for i in data})
                labels.append(data["labels"])
                eval_losses.append(outputs.loss.item())
                predictions.append(outputs.logits)
            
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)    
            eval_loss = np.mean(eval_losses)
        
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