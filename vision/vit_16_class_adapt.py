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

import torch
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ViTConfig, ViTForImageClassification
from transformers import AutoFeatureExtractor
from transformers import Trainer, TrainingArguments

from utils import ImageNet16Class

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    correct = np.sum([predictions[i] == labels[i] for i in range(len(predictions))])
    return {
        "accuracy" : correct/len(predictions)
    }
    
def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def main():

    imnet_16class = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/train")
    imnet_16class_eval = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    #imnet_16class_pooled = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/")

    vit_configs = ViTConfig.from_pretrained("google/vit-base-patch16-224")

    # Change configs to match 16 class dataset
    vit_configs.num_labels = 16
    vit_configs.id2label = {i: imnet_16class.class_names[i] for i in range(16)}
    vit_configs.label2id = {imnet_16class.class_names[i]: i for i in range(16)}

    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    imnet_16class.transform = lambda input: feature_extractor(input, return_tensors="pt")["pixel_values"]
    imnet_16class_eval.transform = lambda input: feature_extractor(input, return_tensors="pt")["pixel_values"]

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=vit_configs, ignore_mismatched_sizes=True)

    # Freeze base model
    for param in model.vit.parameters():
        param.requires_grad = False
    model.vit.eval()

    training_args = TrainingArguments(output_dir="checkpoints/vit_base_16class",
                                    num_train_epochs=5,
                                    learning_rate=2e-3,
                                    evaluation_strategy="epoch",
                                    logging_strategy="epoch",
                                    save_total_limit=1,
                                    save_strategy="epoch",
                                    load_best_model_at_end=True,
                                    per_device_train_batch_size=64,
                                    per_device_eval_batch_size=32,
                                    dataloader_num_workers=4)

    trainer = Trainer(model=model,
                    args=training_args,
                    data_collator=collate_fn,
                    train_dataset=imnet_16class,
                    eval_dataset=imnet_16class_eval,
                    compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("checkpoints/vit_base_16class")
    
if __name__ == "__main__":
    main()