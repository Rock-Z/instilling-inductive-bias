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
import yaml

import torch
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import ResNetModel, ResNetConfig
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import ResNetForImageClassification, ViTForImageClassification

from torchvision import transforms

from utils import ImageNet16Class, ImageNetKaggle
from vision_transfer import load_model

eval_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
                                    lambda x: torch.unsqueeze(x, 0)]
                                )

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

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "label": labels}

def main():
    assert len(sys.argv) == 2, "Usage: python resnet_pooled_imnet_eval.py <checkpoint_path>"
    checkpoint_path = sys.argv[1]

    imnet_16class = ImageNet16Class(root="ImageNet", image_names_dir="16-class-ImageNet/downsampled_1000/eval")
    imnet_eval = filter_to_16_class(ImageNetKaggle(root="ImageNet", split='val', transform=eval_transform), imnet_16class)
    imnet_pooled_eval = filter_to_16_class(ImageNetKaggle(root="MeanPooledImageNet", split='val', transform=eval_transform), imnet_16class)
    
    batch_size = 128
    eval_dataloader = DataLoader(imnet_eval, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    pooled_dataloader = DataLoader(imnet_pooled_eval, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

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

    class_names = imnet_eval.class_names
    sample_size = len(imnet_eval)
    same_prediction = 0
    correct_prediction = 0
    correct_pooled = 0

    sample_size_by_class = {class_name: 0 for class_name in imnet_eval.class_names}
    correct_predictions_by_class = {class_name: 0 for class_name in imnet_eval.class_names}
    correct_predictions_by_class_pooled = {class_name: 0 for class_name in imnet_eval.class_names}

    with torch.no_grad():
        with tqdm(total=sample_size) as pbar:
            for eval_data, pooled_data in zip(eval_dataloader, pooled_dataloader):
                inputs = eval_data['pixel_values'].to(device)
                inputs_pooled = pooled_data['pixel_values'].to(device)
                
                labels = eval_data['label']
                labels_pooled = pooled_data['label']
                
                logits, logits_pooled = model(pixel_values=inputs).logits, model(pixel_values=inputs_pooled).logits
            
                predicted_label = logits.argmax(-1).cpu().numpy()
                predicted_label_pooled = logits_pooled.argmax(-1).cpu().numpy()
                
                for i in range(len(labels)):
                    sample_size_by_class[imnet_eval.class_names[labels[i]]] += 1
                    sample_size += 1
                    
                    if predicted_label[i] == predicted_label_pooled[i]:
                        same_prediction += 1
                    
                    if predicted_label[i] == labels[i]:
                        correct_prediction += 1
                        correct_predictions_by_class[imnet_eval.class_names[labels[i]]] += 1
                    
                    if predicted_label_pooled[i] == labels_pooled[i]:
                        correct_pooled += 1
                        correct_predictions_by_class_pooled[imnet_eval.class_names[labels_pooled[i]]] += 1
                        
                
                #sample_size_by_class[imnet_eval.class_names[imnet_eval[i]['label']]] += 1
                
                #if predicted_label == predicted_label_pooled:
                #    same_prediction += 1
                    
                #if predicted_label == imnet_eval[i]['label']:                
                #    correct_predictions_by_class[imnet_eval.class_names[imnet_eval[i]['label']]] += 1
                    
                #if predicted_label_pooled == imnet_eval[i]['label']:
                #    correct_prediction += 1
                #    correct_predictions_by_class_pooled[imnet_eval.class_names[imnet_eval[i]['label']]] += 1
                    
                pbar.update(batch_size)
                
    acc_original = {class_name: correct_predictions_by_class[class_name]/sample_size_by_class[class_name] for class_name in imnet_eval.class_names}
    acc_pooled = {class_name: correct_predictions_by_class_pooled[class_name]/sample_size_by_class[class_name] for class_name in imnet_eval.class_names}

    #print(f"Fraction of correct predictions: {correct_prediction}/{sample_size} ({correct_prediction/sample_size})")
    print(f"Acc on original: {np.mean(list(acc_original.values()))}")
    print(f"Acc on pooled: {np.mean(list(acc_pooled.values()))}")
    #print(f"Fraction of correct predictions on pooled images: {correct_pooled}/{sample_size} ({correct_pooled/sample_size})")
    #print(f"Fraction of same predictions: {same_prediction}/{sample_size} ({same_prediction/sample_size})")
    print(f"Fraction of correct predictions by class: {acc_original}")
    print(f"Fraction of correct predictions by class for mean pooled images: {acc_pooled}")

"""    plt.figure(figsize=(16, 5))
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

    plt.savefig("resnet18_16class_acc.pdf", bbox_inches='tight')
    plt.savefig("resnet18_16class_acc.png", bbox_inches='tight')"""

if __name__ == "__main__":
    main()