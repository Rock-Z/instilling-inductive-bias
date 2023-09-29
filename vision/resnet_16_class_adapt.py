import sys
import os

# Currently NeuroSurgeon is not a package, add path manually to import
sys.path.append("../")
from NeuroSurgeon.src.Models.circuit_model import CircuitConfig, CircuitModel
from NeuroSurgeon.src.Masking.contsparse_layer import ContSparseLayer

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from tqdm import tqdm
import wandb

import torch
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ResNetModel, ResNetConfig
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from transformers import Trainer, TrainingArguments

from torchvision import transforms

from utils import ImageNet16Class, ImageNetKaggle

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    correct = np.sum([predictions[i] == labels[i] for i in range(len(predictions))])
    return {
        "accuracy" : correct/len(predictions)
    }
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def filter_to_16_class(imnet, imnet_16class):
    # Construct synset to 16-class imnet mapping
    class2synset = {imnet_16class.class_names[i]: [] for i in range(16)}
    synset2class = {}
    class2id = {imnet_16class.class_names[i]: i for i in range(16)}


    for i in range(16):
        class2synset[imnet_16class.class_names[i]] = list(set(map(lambda f: f.split("_")[0], imnet_16class.image_names[i])))
        for synset in class2synset[imnet_16class.class_names[i]]:
            synset2class[synset] = imnet_16class.class_names[i]

    # Convert ImNet1k to 16-class ImNet
    for i in range(len(imnet)):
        synset = imnet.val_to_syn[imnet.samples[i].split('/')[-1]]
        if synset in synset2class:
            new_target = class2id[synset2class[synset]]
            imnet.targets[i] = new_target
        else:
            imnet.targets[i] = -1

    # Filer out all invalid targets
    imnet.samples = [imnet.samples[i] for i in range(len(imnet.samples)) if imnet.targets[i] != -1]   
    imnet.targets = [imnet.targets[i] for i in range(len(imnet.targets)) if imnet.targets[i] != -1]        

    imnet.class_names = imnet_16class.class_names
    
    return imnet

def main():
    # WandB logging
    wandb.login()

    run = wandb.init(
        project= "resnet_augmentation_baseline"
    )
    
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

    imnet_16class = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet", transform=train_transform)
    imnet_16class_eval = filter_to_16_class(ImageNetKaggle(root="ImageNet", split="val", transform=eval_transform), imnet_16class)
    #imnet_16class_eval = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    imnet_16class_pooled = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/train", transform=train_transform)
    imnet_16class_pooled_eval = filter_to_16_class(ImageNetKaggle(root="MeanPooledImageNet", split="val", transform=eval_transform), imnet_16class)
    #imnet_16class_pooled_eval = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    
    # Combine two training sets as data augmentation
    imnet_16class.samples.append(imnet_16class_pooled.samples)
    imnet_16class.targets.append(imnet_16class_pooled.targets)
    imnet_16class.build_dataset()
    train_set = imnet_16class

    resnet_configs = ResNetConfig.from_pretrained("microsoft/resnet-18")

    # Change configs to match 16 class dataset
    resnet_configs.num_labels = 16
    resnet_configs.id2label = {i: imnet_16class.class_names[i] for i in range(16)}
    resnet_configs.label2id = {imnet_16class.class_names[i]: i for i in range(16)}

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18", config=resnet_configs, ignore_mismatched_sizes=True)

    # Freeze base model
    #for param in model.resnet.parameters():
    #    param.requires_grad = False
    #model.resnet.eval()

    training_args = TrainingArguments(output_dir="checkpoints/resnet18_16class_ft_augmentation",
                                    num_train_epochs=5,
                                    learning_rate=1e-3,
                                    evaluation_strategy="epoch",
                                    logging_strategy="epoch",
                                    save_total_limit=1,
                                    save_strategy="epoch",
                                    metric_for_best_model="eval_original_accuracy",
                                    load_best_model_at_end=True,
                                    per_device_train_batch_size=128,
                                    per_device_eval_batch_size=128,
                                    dataloader_num_workers=4)

    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train_set,
                    eval_dataset={"original":imnet_16class_eval, "pooled":imnet_16class_pooled_eval},
                    compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("checkpoints/resnet18_16class_ft_augmentation")
    
if __name__ == "__main__":
    main()