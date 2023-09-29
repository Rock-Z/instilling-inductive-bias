import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback

import os
import shutil
import sys
import yaml

from utils import generate_samples, set_random_seed, format, Numbers        
        
def main():
    # Load configuration
    assert len(sys.argv) == 2, "Usage: main.py <config.yaml>"

    with open(sys.argv[1], 'r') as file:
        args = yaml.safe_load(file)
        reproducibility_configs = args["reproducibility_configs"]
        data_configs = args["data_configs"]
        trainer_configs = args["trainer_configs"]
        GPT_configs = args["GPT_configs"]    
        
    # Copy config file to output path 
    os.makedirs(trainer_configs['output_dir'], exist_ok=True)
    shutil.copyfile(sys.argv[1], trainer_configs["output_dir"] + '/configs.yaml')
    
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
        with open(trainer_configs["output_dir"] + "/seed.txt", "w") as f:
            f.write(str(seed))  
    
    # Generate training data
    X= generate_samples(**data_configs, computation="a + ab")
    train_dataset = Numbers(X)

    # Always eval using 1000 samples
    data_configs["n_samples"] = 1000
    X_eval= generate_samples(**data_configs, computation="a + ab")
    eval_dataset = Numbers(X_eval)

    # Create model
    gpt_configs = GPT2Config(**GPT_configs)

    model = GPT2LMHeadModel(gpt_configs)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[:, -2, :], axis=1)
        correct = np.sum([predictions[i] == labels[i, -1] for i in range(len(predictions))])
        return {
            "accuracy" : correct/len(predictions)
        }

    training_args = TrainingArguments(**trainer_configs)

    trainer = Trainer(model = model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
                    )

    trainer.train()
    trainer.save_model(trainer_configs["output_dir"])

if __name__ == "__main__":
    main()