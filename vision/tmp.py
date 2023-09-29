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
from transformers import ResNetModel, ResNetConfig
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from utils import ImageNet16Class

from vision_ablate import load_model

train_configs = {"model_path": "checkpoints/resnet18_16class"}

circuit_configs = { "mask_method": "continuous_sparsification",
                    "mask_hparams":{
                        "ablation": "none",
                        "mask_unit": "weight",
                        "mask_bias": False,
                        "mask_init_value": -0.1},
                    "freeze_base" : True,
                    "add_l0" : True,
                    "l0_lambda" : 1.0e-6}

def main():

    imnet_16class = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    imnet_16class_pooled = ImageNet16Class(root="MeanPooledImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")

    feature_extractor= AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    
    model, _ = load_model(train_configs, circuit_configs)
    model.init_subnet_transfer()
    circuit_state_dict = torch.load("checkpoints/resnet18_1e-6_transfer/latest_circuit_model.pt")
    model.load_state_dict(circuit_state_dict)
    model.to("cuda")
    model.eval()

    sample_size = len(imnet_16class)
    same_prediction = 0
    correct_prediction = 0

    sample_size_by_class = {class_name: 0 for class_name in imnet_16class.class_names}
    correct_predictions_by_class = {class_name: 0 for class_name in imnet_16class.class_names}
    correct_predictions_by_class_pooled = {class_name: 0 for class_name in imnet_16class.class_names}

    with torch.no_grad():
        with tqdm(total=sample_size) as pbar:
            for i in range(len(imnet_16class)):
                inputs = feature_extractor(imnet_16class[i]['img'], return_tensors="pt").to("cuda")
                inputs_pooled = feature_extractor(imnet_16class_pooled[i]['img'], return_tensors="pt").to("cuda")
                logits, logits_pooled = model(**inputs).logits, model(**inputs_pooled).logits
            
                predicted_label = logits.argmax(-1).item()
                predicted_label_pooled = logits_pooled.argmax(-1).item()
                
                sample_size_by_class[imnet_16class.class_names[imnet_16class[i]['label']]] += 1
                
                if predicted_label == predicted_label_pooled:
                    same_prediction += 1
                    
                if predicted_label == imnet_16class[i]['label']:                
                    correct_predictions_by_class[imnet_16class.class_names[imnet_16class[i]['label']]] += 1
                    
                if predicted_label_pooled == imnet_16class[i]['label']:
                    correct_prediction += 1
                    correct_predictions_by_class_pooled[imnet_16class.class_names[imnet_16class[i]['label']]] += 1
                    
                pbar.update(1)
                
    acc_original = {class_name: correct_predictions_by_class[class_name]/sample_size_by_class[class_name] for class_name in imnet_16class.class_names}
    acc_pooled = {class_name: correct_predictions_by_class_pooled[class_name]/sample_size_by_class[class_name] for class_name in imnet_16class.class_names}

    print(f"Fraction of same predictions: {same_prediction}/{sample_size} ({same_prediction/sample_size})")
    print(f"Fraction of correct predictions: {correct_prediction}/{sample_size} ({correct_prediction/sample_size})")
    print(f"Fraction of correct predictions by class: {acc_original}")
    print(f"Fraction of correct predictions by class for mean pooled images: {acc_pooled}")

    plt.figure(figsize=(16, 5))
    plt.bar(np.arange(16), list(acc_pooled.values()), width=0.3, label="Mean Pooled Images", color='darkturquoise')
    plt.bar(np.arange(16)+0.3, list(acc_original.values()), width=0.3, label="Original Images", color='sandybrown')
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.xticks(ticks=np.arange(16)+0.3, labels=acc_pooled.keys(), rotation=-45)
    plt.title("ResNet18 Accuracy on 16 Class ImageNet, Per Class")
    plt.axline((0, 1/16), (8, 1/16), color='k', linestyle='--', linewidth=1, label="Chance")
    plt.axline((0, sum(acc_pooled.values())/16), (8, sum(acc_pooled.values())/16), color='darkturquoise', linestyle='--', linewidth=1, label="Avg. Mean Pooled")
    plt.axline((0, sum(acc_original.values())/16), (8, sum(acc_original.values())/16), color='sandybrown', linestyle='--', linewidth=1, label="Avg. Original")
    plt.legend()

    plt.savefig("resnet18_16class_acc.svg", bbox_inches='tight')
    plt.savefig("resnet18_16class_acc.png", bbox_inches='tight')

if __name__ == "__main__":
    main()