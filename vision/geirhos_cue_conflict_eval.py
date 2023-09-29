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
import re

import cv2
import torch
from torch import optim
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ResNetModel, ResNetConfig, ViTForImageClassification
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import Trainer, TrainingArguments

from utils import ImageNet16Class, format
from vision_transfer import load_model

from torchvision import transforms

def main():
    assert len(sys.argv) == 2, "Usage: geirhos_cue_conflict_eval.py <path_to_model>"
    
    checkpoint_path = sys.argv[1]
    
    try:
        with open(os.path.join(checkpoint_path, "configs.yaml"), "r") as f:
            configs = yaml.safe_load(f)
        
        # Load model by first loading a model of same architecture from config file and then copying state dict
        model, device = load_model(configs["train_configs"], configs["circuit_configs"])
        state_dict = torch.load(os.path.join(checkpoint_path, "best_circuit_model.pt"))
        
        # Manually handle ViT state dict entry names
        if isinstance(model, ViTForImageClassification):
            new_state_dict = {}
            for entry in state_dict:
                new_key = entry.split(".", 1)[-1]
                new_state_dict[new_key] = state_dict[entry]  
            
            state_dict = new_state_dict                  
        
        model.load_state_dict(state_dict)
        #model = ResNetForImageClassification.from_pretrained(checkpoint_path)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        model.eval()
    except:
        model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    
    feature_extractor = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(224), 
                                            transforms.CenterCrop(224),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    #feature_extractor = AutoFeatureExtractor.from_pretrained(configs["train_configs"]["hf_pretrained"])
    
    class_correct = {}
    class_total = {}
    class_misled = {}
    
    geirhos_cue_conflict = "texture-vs-shape/stimuli/style-transfer-preprocessed-512"
    if isinstance(model, CircuitModel):        
        id2label = model.root_model.config.id2label
        label2id = model.root_model.config.label2id
    else:
        id2label = model.config.id2label
        label2id = model.config.label2id
        
    print(id2label)
    # Assume the geirhos texture-vs-shape repo is in the vision folder
    for imnet_class in os.listdir(geirhos_cue_conflict):
        class_correct[imnet_class] = 0
        class_total[imnet_class] = 0
        class_misled[imnet_class] = 0
    
    
    for imnet_class in os.listdir(geirhos_cue_conflict):
        
        gt_label = label2id[imnet_class]
        
        for image in os.listdir(os.path.join(geirhos_cue_conflict, imnet_class)):            
            img = cv2.imread(os.path.join(geirhos_cue_conflict, imnet_class, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocessed = feature_extractor(img)
            preprocessed = torch.unsqueeze(preprocessed, 0)
            
            # Find the misleading texture label from image name
            for im_cls in label2id:
                if im_cls in image and im_cls != imnet_class:
                    texture_label = label2id[im_cls]
            
            with torch.no_grad():
                logits = model(pixel_values=preprocessed.to(device)).logits
                prediction = torch.argmax(logits, dim=1).item()
            
            class_correct[imnet_class] += int(prediction == gt_label)
            class_misled[id2label[texture_label]] += int(prediction == texture_label)
            class_total[imnet_class] += 1
            
            print(f"image name = {image}, predicted = {id2label[prediction]}, gt = {id2label[gt_label]}")

    accuracy = {class_name: class_correct[class_name] / class_total[class_name] for class_name in class_correct}
    accuracy_mislead = {class_name: class_misled[class_name] / class_total[class_name] for class_name in class_misled}
    print("Accuracy: ", accuracy)
    print("Misled:", accuracy_mislead)
    print("average accuracy", np.mean(list(accuracy.values())))
    print("average misled", np.mean(list(accuracy_mislead.values())))

if __name__ == "__main__":
    main()