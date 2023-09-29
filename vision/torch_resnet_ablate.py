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

import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.models import resnet50, resnet18
from torchvision.models.resnet import ResNet, BasicBlock

import wandb
WANDB = True

resnet18_model = resnet18(weights = 'DEFAULT')
resnet18_model.eval()

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def enumerate_BSDS(root, sub_dir_name = 'train'):
    PATH = os.path.join(root, 'BSDS500/data/')
    assert sub_dir_name in ['train', 'test', 'val']
    img_pth = os.path.join(PATH, 'images', sub_dir_name)
    gt_path = os.path.join(PATH, 'groundTruth', sub_dir_name)
    for index in range(len(os.listdir(img_pth))):
        filename = os.listdir(img_pth)[index]
        # BSDS500 image folders seem to have Thmubs.db in them, exlucde these
        if not ("Thumbs" in filename):
            annotations = io.loadmat(os.path.join(gt_path, filename.split('.')[0]+'.mat'))['groundTruth'][0]
            outlines = [annotations[i][0][0][1] for i in range(len(annotations))]
            yield {"img_path": os.path.join(img_pth, filename), "annotations": outlines}
            
read_image_as_float = lambda path: read_image(path).float() / 255.0
            
transforms = T.Compose([T.Resize(256),
                        T.RandomCrop(224), 
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def compute_similarity_target(idx):
    return (np.broadcast_to(idx, (len(idx), len(idx))) == np.broadcast_to(idx, (len(idx), len(idx))).T).astype(np.float32)

class BSDS(Dataset):
    def __init__(self, dataset_enumerator) -> None:
        super().__init__()
        
        X = []
        y = []
        n_annotators = []
        
        for data in dataset_enumerator:
            img = read_image_as_float(data['img_path'])
            X.append(img)
            y += [torch.Tensor(i) for i in data['annotations']]
            n_annotators.append(len(data['annotations']))
            
        self.X = X
        self.y = y
        self.n_annotators = n_annotators
        self.dataset_idx_to_image_idx = np.repeat(np.arange(len(self.X)), self.n_annotators)
        
        self._len = sum(n_annotators)
        
    def __len__(self) -> int:
        return self._len
    
    def transform(self, X, y):
        """Transform a single pair of X and y raw data into the expected input format of ResNet"""
        resize = T.Resize(256)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        X = resize(X)
        y = resize(y.repeat(3, 1, 1))
        
        i, j, h, w = T.RandomCrop.get_params(X, output_size=(224, 224))
        X = T.functional.crop(X, i, j, h, w)
        y = T.functional.crop(y, i, j, h, w)
        
        #X, y = normalize(X), normalize(y)
                
        return (X, y)
    
    def __getitem__(self, index):        
        x_idx = self.dataset_idx_to_image_idx[index]
        return (*self.transform(self.X[x_idx], self.y[index]), x_idx)
    
    
class ResNetForRepresentationMatching(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove last layer
        del self.fc
        
    def from_resnet18(resnet18_model):
        model = ResNetForRepresentationMatching(BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(resnet18_model.state_dict(), strict=False)
        return model
    
    def compute_representation(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # 112 x 112
        
        x = self.maxpool(x) # 56 x 56
        
        x = self.layer1(x) # 56 x 56
        x = self.layer2(x) # 28 x 28
        x = self.layer3(x) # 14 x 14
        x = self.layer4(x) # 7 x 7
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, image, annotation = None, idx = None, return_dict = True):
        image_rep = self.compute_representation(image)        
        annotation_rep = self.compute_representation(annotation)
        
        similarity_target = torch.tensor(compute_similarity_target(idx)).to(image_rep.device)
        
        # Compute cosine similarity between batches
        i_n, a_n = image_rep.norm(dim = 1)[:, None], annotation_rep.norm(dim = 1)[:, None]
        image_rep_norm = image_rep / torch.clamp(i_n, min = 1e-8)
        annotation_rep_norm = annotation_rep / torch.clamp(a_n, min = 1e-8)
        
        image_annotation_similarity = torch.mm(image_rep_norm, annotation_rep_norm.transpose(0, 1))
        image_image_similarity = torch.mm(image_rep_norm, image_rep_norm.transpose(0, 1))
        
        loss_fn = torch.nn.KLDivLoss()
        
        loss = loss_fn(image_annotation_similarity, similarity_target) + loss_fn(image_image_similarity, similarity_target)
        
        if return_dict:
            return {"image_rep": image_rep, "annotation_rep": annotation_rep, "similarity": image_annotation_similarity, "loss": loss}        
        else:
            return image_rep

def compute_metrics(eval_pred, circuit_model):
    """Given evaluation prediction tuple, compute metris and return as dict

    Args:
        eval_pred (tuple(Tensor, Tensor)): tuple of prediction Tensor of shape (batch_size, seq_len, vocab_size) and 
        label of shape (batch_size, seq_len)

    Returns:
        dict: Dictionary containing prediction accuracy (computed with label of last token only), along with 
        ratio of nonzero params and mean of mask weights
    """
    image_rep, annotations_rep, idx = eval_pred
    
    with torch.no_grad():
        image_annotation_similarity = torch.mm(image_rep, annotation_rep.transpose(0, 1))
        image_image_similarity = torch.mm(image_rep, image_rep.transpose(0, 1))            
        target = torch.tensor(compute_similarity_target(idx))
            
        eval_loss = loss(image_annotation_similarity, target) + loss(image_image_similarity, target)
    
        im_anno_similarity_positive = torch.mean(image_annotation_similarity * torch.eye(len(image_annotation_similarity)))
        im_anno_similarity_negative = torch.mean(image_annotation_similarity * (1 - torch.eye(len(image_annotation_similarity))))
    
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
        "eval_loss": eval_loss,    
        "mean_similarity_positive": im_anno_similarity_positive,
        "mean_similarity_negative": im_anno_similarity_negative,
        # Strictly speaking this is not exactly "mean", as # of params in each layer is different
        "mask_param_mean": round(mean / n_modules, 4), 
        "nonzero_params_percentage": round(nonzero_params / total_params, 4),
    }

model = ResNetForRepresentationMatching.from_resnet18(resnet18_model)

circuit_configs = {
    "mask_method": "continuous_sparsification",
    "mask_hparams": {
        "ablation": "none",
        "mask_unit": "weight",
        "mask_bias": False,
        "mask_init_value": -0.1
        },
    "freeze_base" : True,
    "add_l0" : True
    }

train_configs = {
    "n_epochs": 100,
    "final_temp": 100
    }

layers_to_ablate = []

for name, module in model.named_modules():
    if isinstance(module, Conv2d):
        layers_to_ablate.append(name)

circuit_configs["target_layers"] = layers_to_ablate

circuit_model = CircuitModel(CircuitConfig(**circuit_configs), model)

# Load BSDS dataset
dataset = BSDS(enumerate_BSDS("BSR"))
dataset_val = BSDS(enumerate_BSDS("BSR", 'val'))

dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
dataloader_val = DataLoader(dataset_val, batch_size = 8, shuffle = True)

optimizer = torch.optim.SGD(circuit_model.parameters(), lr = 1e-3, momentum = 0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
circuit_model.to(device)

loss = torch.nn.KLDivLoss()

if WANDB:
    # WandB logging
    wandb.login()

    run = wandb.init(
        project= "vision-ablation",
    )

temp = 1

for epoch in range(train_configs["n_epochs"]):
    circuit_model.train(True)
    
    running_loss = 0.
    last_loss = 0.
    
    with tqdm(total = len(dataloader)) as pbar:
        pbar.set_description(f"epoch {epoch + 1} batch 0 loss: {last_loss}")
        pbar.set_postfix({"temp": round(temp, 2)})
        for i, data in enumerate(dataloader):
            
            images, annotations, idx = data
            outputs = circuit_model(image = images.to(device), annotation = annotations.to(device), idx = idx)
            
            loss = outputs["loss"]
            loss.backward()
            
            # Train!
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            
            running_loss += loss.item()
            # Update data 10 times per epoch
            if i % (len(dataloader) // 10) == (len(dataloader) // 10 - 1):
                last_loss = running_loss / (len(dataloader) // 10)
                pbar.set_description(f"Epoch {epoch + 1} batch {i + 1} loss: {round(last_loss, 4)}")
                running_loss = 0.
        
    # Evaluate at epoch end
    circuit_model.eval()    
    with torch.no_grad():        
        for i, data in enumerate(dataloader_val):
            image, annotation, idx = data
            image_rep = torch.norm(circuit_model(x = image.to(device)), dim = 1)
            annotation_rep = torch.norm(circuit_model(x = annotation.to(device)), dim = 1)
            target = torch.tensor(compute_similarity_target(idx))
            break
        
        eval_stats = compute_metrics((image_rep, annotation_rep, target), circuit_model)
    
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
    print(format(f"Temp increased to {temp}", "BLUE"))
