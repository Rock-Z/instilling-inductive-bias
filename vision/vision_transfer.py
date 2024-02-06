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
from torch.nn import Conv2d, Linear, DataParallel
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ResNetModel, ResNetConfig, AutoConfig
from transformers import ViTForImageClassification, ResNetForImageClassification, AutoModelForImageClassification
from transformers import Trainer, TrainingArguments

from utils import ImageNet16Class, format

from torchvision import transforms

WANDB = True

def set_random_seed(seed : int):
    """Fix torch and numpy seed for run reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)    

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def load_model(train_configs, circuit_configs):
    # Load pretrained ResNet previous adapted to ImageNet16Class
    # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    
    # Intercept and train from random initialization if inherit from is specified to random init.
    if "inherit_from" in train_configs:
        if train_configs["inherit_from"] == "random":
            config = AutoConfig.from_pretrained(train_configs["hf_pretrained"])
            # Change configs to match 16 class dataset
            imnet_16class = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet")
            config.num_labels = 16
            config.id2label = {i: imnet_16class.class_names[i] for i in range(16)}
            config.label2id = {imnet_16class.class_names[i]: i for i in range(16)}
            
            model = AutoModelForImageClassification.from_config(config)
            device = torch.device("cuda")
            model.to(device)
            model.train()
            
            return model, device            
    
    # Load 16 class adapted model
    model = AutoModelForImageClassification.from_pretrained(train_configs["original"])
    device = torch.device("cuda")
    
    # Create CircuitModel for masking
    if isinstance(model, ViTForImageClassification):
        layers_to_ablate = list(map(lambda i: i[0], filter(lambda i: isinstance(i[1], Conv2d) or isinstance(i[1], Linear), model.named_modules())))
    elif isinstance(model, ResNetForImageClassification):
        raise ValueError("Freezing embeds only work for ViT")
        layers_to_ablate = list(map(lambda i: i[0],filter(lambda i: isinstance(i[1], Conv2d), model.named_modules())))
    else:        
        # Raise error if model is not supported
        raise NotImplementedError(f"Model {type(model)} not supported")
        
    circuit_config = CircuitConfig(
        target_layers=layers_to_ablate,
        **circuit_configs
        )    
    circuit_model = CircuitModel(circuit_config, model)
    
    subnet_state_dict = torch.load(train_configs["subnet"] + "/best_circuit_model.pt")
    circuit_model.load_state_dict(subnet_state_dict, strict=False)
    circuit_model.init_subnet_transfer()
    
    # Reset regular params
    for module in circuit_model.modules():
        if isinstance(module, ContSparseLayer):
            module.weight.requires_grad = True
            if hasattr(module, "bias") and module.bias != None:
                module.bias.requires_grad = True
    
    # Freeze CLS token
        circuit_model.root_model.vit.embeddings.cls_token.requires_grad = False
    
    circuit_model.to(device)
    circuit_model.train()
    
    return circuit_model, device

def compute_metrics(eval_pred):
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
    
    return {
        "accuracy" : correct/len(predictions)
    }
    
train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(256),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(224),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])]
)

eval_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])]
                                    )

def schedule_lr(optimizer, lr : float, lr_scheduler_configs : dict, step_ct : int, total_steps : int):
    if "warmup" in lr_scheduler_configs:
        if step_ct < lr_scheduler_configs["warmup_steps"]:
            for g in optimizer.param_groups:
                g['lr'] = lr * step_ct / lr_scheduler_configs["warmup_steps"]
    if "decay" in lr_scheduler_configs:
        warmup = lr_scheduler_configs["warmup_steps"] if "warmup_steps" in lr_scheduler_configs else 0
        if step_ct > warmup:
            for g in optimizer.param_groups:
                g['lr'] = lr * (1 - (step_ct - warmup) / (total_steps - warmup))

def main():
    # Load configuration
    assert len(sys.argv) == 2, "Usage: resnet_transfer.py <config.yaml>"

    with open(sys.argv[1], 'r') as file:
        args = yaml.safe_load(file)
        reproducibility_configs = args["reproducibility_configs"]
        dataloader_configs = args["dataloader_configs"]
        circuit_configs = args["circuit_configs"]
        train_configs = args["train_configs"]
        optim_configs = args["optim_configs"]
        lr_scheduler_configs = None if "lr_scheduler_configs" not in args else args["lr_scheduler_configs"]
        
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
    
    train_set = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet")
    
    eval_set = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    eval_set_pooled = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    
    # Remove overlap between train and eval set
    for cls_idx, img_cls in enumerate(train_set.image_names):
        for img in img_cls:
            if img in eval_set.image_names[cls_idx]:
                train_set.image_names[cls_idx].remove(img)
    # re-build dataset
    train_set.build_dataset()
    
    # Load feature extractor into dataset
    # feature_extractor = AutoFeatureExtractor.from_pretrained(hf_pretrained)
    train_set.transform = train_transform
    eval_set.transform = eval_transform
    eval_set_pooled.transform = eval_transform
    
    train_dataloader = DataLoader(train_set, collate_fn=collate_fn, **dataloader_configs)
    eval_dataloader = DataLoader(eval_set, collate_fn=collate_fn,**dataloader_configs)
    eval_dataloader_pooled = DataLoader(eval_set_pooled, collate_fn=collate_fn,**dataloader_configs)

    circuit_model, device = load_model(train_configs, circuit_configs)

    # Print out all trainable parameters as sanity check
    if verbose:
        for name, param in circuit_model.named_parameters():
            if param.requires_grad:
                print(format("With Grad:", "BOLD"), name)
            else:
                print(format("No Grad:", "BOLD"), name)
    
    if torch.cuda.device_count() > 1:
        print(format(f"Using {torch.cuda.device_count()} GPUs", "BOLD"))
        circuit_model = DataParallel(circuit_model)
    
    optimizer = getattr(optim, optim_configs["optim_name"])\
        (circuit_model.parameters(), **{i:optim_configs[i] for i in optim_configs if i != "optim_name"})
        
    if verbose:
        print(format("Initialized optimizer:\n", "BOLD"), optimizer)

    if WANDB:
        # WandB logging
        wandb.login()

        run = wandb.init(
            project= "subnet-transfer" if "wandb_project" not in train_configs else train_configs["wandb_project"],
            config = args
        )
    
    # Weight each class by inverse of its frequency in dataset since 16class-ImageNet is imbalanced
    loss_weight = torch.tensor([len(train_set)/len(img_cls) for img_cls in train_set.image_names])
    loss_weight /= loss_weight.norm()
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight.to(device))

    best_acc = 0
    step_ct = 0
    total_steps = train_configs["n_epochs"] * len(train_dataloader)
    for epoch in range(train_configs["n_epochs"]):
        circuit_model.train(True)
        
        running_loss = 0.
        last_loss = 0.
        
        with tqdm(total = len(train_dataloader)) as pbar:
            pbar.set_description(f"epoch {epoch + 1} batch 0 loss: {last_loss}, lr={optimizer.param_groups[0]['lr']}")
            for i, data in enumerate(train_dataloader):
                # Need to override default loss, thus only pass in pixel values
                outputs = circuit_model(pixel_values=data["pixel_values"].to(device))
                
                loss = loss_fn(outputs.logits, data["labels"].to(device))        
                running_loss += loss.item()
                loss.backward()
                
                # Train!
                optimizer.step()
                optimizer.zero_grad()
                step_ct += 1
                pbar.update(1)
                
                # Schedule learning rate if scheduler is specified
                if lr_scheduler_configs is not None:
                    schedule_lr(optimizer, optim_configs['lr'], lr_scheduler_configs, step_ct, total_steps)
                
                # Update data 100 times per epoch
                if i % (len(train_dataloader) // 100) == (len(train_dataloader) // 100 - 1):
                    last_loss = running_loss / (len(train_dataloader) // 100)
                    pbar.set_description(f"Epoch {epoch + 1} batch {i + 1} loss: {round(last_loss, 4)}")
                    running_loss = 0.
            
                # Evaluate 5 times per epoch
                if i % (len(train_dataloader) // 5) == (len(train_dataloader) // 5 - 1):
                    circuit_model.eval()    
                    with torch.no_grad():    
                        predictions = []
                        predictions_pooled = []
                        labels = []
                        labels_pooled = []
                        
                        for data, data_pooled in zip(eval_dataloader, eval_dataloader_pooled):    
                            outputs = circuit_model(pixel_values=data["pixel_values"].to(device))
                            outputs_pooled = circuit_model(pixel_values=data_pooled["pixel_values"].to(device))
                            labels.append(data["labels"])
                            labels_pooled.append(data_pooled["labels"])
                            predictions.append(outputs.logits)
                            predictions_pooled.append(outputs_pooled.logits)
                        
                        predictions = torch.cat(predictions, dim=0)
                        predictions_pooled = torch.cat(predictions_pooled, dim=0)
                        labels = torch.cat(labels, dim=0)    
                        labels_pooled = torch.cat(labels_pooled, dim=0)

                        eval_stats = compute_metrics((predictions, labels))
                        eval_stats_pooled = compute_metrics((predictions_pooled, labels_pooled))
                        eval_stats["pooled_acc"] = eval_stats_pooled["accuracy"]
                        eval_acc = eval_stats["accuracy"]
                    
                    # Send log data to wandb
                    if WANDB:
                        eval_stats["last_train_loss"] = last_loss
                        eval_stats["step"] = step_ct
                        eval_stats["epoch"] = epoch
                        wandb.log(eval_stats)
        
                    # Print eval stats to terminal
                    print(format(f"Eval stats at step {step_ct} in epoch {epoch}: {eval_stats}", "YELLOW"))
                
        # Save model
        torch.save(circuit_model.state_dict(), train_configs["output_path"] + f"/latest_circuit_model.pt")
        if eval_acc >= best_acc:
            best_acc = eval_acc
            print(format(f"Saving model with eval acc {eval_acc}", "BLUE"))
            torch.save(circuit_model.state_dict(), train_configs["output_path"] + f"/best_circuit_model.pt")
    
if __name__ == "__main__":
    main()