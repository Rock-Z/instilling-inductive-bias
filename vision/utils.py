import torch
from torch.utils.data import Dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np

import json
import os
import random

class ImageNet16Class(Dataset):
    """Build a 16-class ImageNet dataset with the Kaggle ImageNet from Geihros' et. al.'s setup.
    """
    def __init__(self, root, image_names_dir, transform=None):
        super().__init__()
        
        self.root = root
        self.transform = transform
        self.class_names = []
        self.image_names = [[] for i in range(16)]
        
        for class_txt in os.listdir(os.path.join(image_names_dir, "image_names")):
            class_name = class_txt.split(".")[0]
            class_index = len(self.class_names)
            self.class_names.append(class_name)
            
            with open(os.path.join(os.path.join(image_names_dir, "image_names"), class_txt), "r") as f:
                for line in f:
                    self.image_names[class_index].append(line.strip())
        
        self.build_dataset()
                    
    def build_dataset(self):
        self.samples = []
        self.targets = []
            
        samples_dir = os.path.join(self.root, "ILSVRC/Data/CLS-LOC/train")
        
        for class_index in range(len(self.class_names)):
            for file_name in self.image_names[class_index]:
                sample_path = os.path.join(samples_dir, file_name.split("_")[0], file_name)
                self.samples.append(sample_path)
                self.targets.append(class_index)
                
    def downsample(self, max_sample_per_class):
        """Downsample dataset so it only contains max_sample_per_class samples per class.

        Args:
            max_sample_per_class (int): Classes with more samples than this number will be
            downsampled to contain only this number. Classes with less samples will not be affected
        """
        
        new_img_names = []
        
        for images in self.image_names:
            if len(images) > max_sample_per_class:
                random.shuffle(images)
                new_img_names.append(images[:max_sample_per_class])
            else:
                new_img_names.append(images)
                
        self.image_names = new_img_names
        # Re-build the dataset from new set of images
        self.build_dataset()        
        
    def export_image_names(self, image_names_dir):
        """Export images contained in dataset to txt files in the 16-class-ImageNet format, for
        loading back data in the future

        Args:
            image_names_dir (str): folder to export txt files to
        """
        
        parent_dir = os.path.join(image_names_dir, "image_names")
        os.makedirs(parent_dir, exist_ok = True)
        
        for class_index, images in enumerate(self.image_names):
            with open(os.path.join(parent_dir, f"{self.class_names[class_index]}.txt"), "w") as f:
                for image in images:
                    f.write(image + "\n")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x = cv2.imread(self.samples[idx])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if self.transform:
            pixel_values = self.transform(x)            
            return {"img": x, "pixel_values": pixel_values, "label": self.targets[idx]}
        else:
            return {"img": x, "label": self.targets[idx]}


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.syn_to_class_name = {}
        self.class_names = [0] * 1000
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
                self.syn_to_class_name[v[0]] = v[1]
                self.class_names[int(class_id)] = v[1]
                        
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
                    
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x = cv2.imread(self.samples[idx])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        #x = torchvision.io.read_image(self.samples[idx])
        if self.transform:
            pixel_values = self.transform(x)
            return {"img": x, "pixel_values": pixel_values, "label": self.targets[idx]}
        else:
            return {"img": x, "label": self.targets[idx]}

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.3]])
        img[m] = color_mask
    ax.imshow(img)
    
def mean_pool(image, anns):
    output_image = image.copy()
    # Complement is the complemente region of *all* masks, i.e. the parts of image with no masks on it
    # Used to make sure all texture are ablated even if they are not covered by any mask
    complement = np.ones(image.shape[:2], dtype=bool)
    
    # Sort mask by descending area, so smaller masks take priority since patches
    # for them are created last
    anns.sort(key=lambda x: x['area'], reverse=True)
    
    for ann in anns:
        # Extract mask region from dict
        segmentation = ann['segmentation']
        # Set mask region of image as mean of region
        output_image[segmentation] = image[segmentation].mean(axis=0)
        # Keep track of regions that has no mask on it
        complement = np.logical_and(complement, np.logical_not(segmentation))
    
    # Finally set complement region as mean of region
    output_image[complement] = image[complement].mean(axis=0)
    
    return output_image

# Helper class for formatting debug printouts
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'   
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

# Colorful (& bold) text! 
format = lambda text, style: getattr(Color, style) + text + Color.END