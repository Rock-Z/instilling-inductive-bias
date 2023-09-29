import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback

# Define upper & lower bound for math task
max = 1000

def generate_samples(n_samples : int, max: int, p: int, ambiguous : bool = True, computation: str = "a^2 + ab"):
    """
    Generate training samples for the calculation of (a^2 + ab) % p 
    """
    
    
    # Initialize return variables
    X = np.zeros(shape=(n_samples, 4), dtype=int)
    
    assert n_samples >= max, "Must have at least enough samples to cover all numbers"
    
    if ambiguous:
        assert n_samples <= max, "If ambiguous, can only have sample size == # of different numbers since all numbers need to be covered"
        # The training samples always have the same a & b, so it's ambiguous what the computation is
        X[:, 0:2] = np.stack([np.arange(max)] * 2).T
    else:    
        # To cover all numbers, one of a and b are randomly chosen to be each of 1 to max, and the other
        # is randomly generated
        position = np.random.randint(2, size=max)
        X[np.arange(max), position] = np.arange(max)
        X[np.arange(max), 1 - position] = np.random.randint(1, max, size=max)
    
        # Generate rest of the samples truly randomly
        X[max:, :2] = np.random.randint(max, size= (n_samples - max, 2))
        
        
    # Third column is special token
    X[:, 2] = max
    # Fourth column is the answer
    assert computation in ["a^2 + ab", "2a^2", "2b^2"]
    if computation == "a^2 + ab":
        X[:, 3] = np.mod(np.square(X[:, 0]) + X[:, 0] * X[:, 1], p)
    elif computation == "2a^2":
        X[:, 3] = np.mod(2 * np.square(X[:, 0]), p)
    elif computation == "2b^2":
        X[:, 3] = np.mod(2 * np.square(X[:, 1]), p)
    
    return X

class Numbers(Dataset):
    def __init__(self, X) -> None:
        super().__init__()
        
        self.X = torch.from_numpy(X)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        return {'input_ids':x, 'labels':x}
    
    def to_cuda(self):
        self.X = self.X.cuda()
        self.y = self.y.cuda()
        
        
# Generate training data

X= generate_samples(max, max, 7)
train_dataset = Numbers(X)

X_eval = generate_samples(1000, max, 7)
X_eval_ab= generate_samples(1000, max, 7, ambiguous=False, computation= "a^2 + ab")
X_eval_a= generate_samples(1000, max, 7, ambiguous=False, computation= "2a^2")
X_eval_b = generate_samples(1000, max, 7, ambiguous=False, computation= "2b^2")

eval, eval_ab, eval_a, eval_b = Numbers(X_eval), Numbers(X_eval_ab), Numbers(X_eval_a), Numbers(X_eval_b)


# Create model

configs = GPT2Config(vocab_size=max + 1, 
                     n_positions=4, 
                     n_embd=64,
                     n_layer=12, 
                     n_head=4, 
                     resid_pdrop=0.1, 
                     embd_pdrop=0.1)

model = GPT2LMHeadModel(configs)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[:, -2, :], axis=1)
    correct = np.sum([predictions[i] == labels[i, -1] for i in range(len(predictions))])
    return {
        "accuracy" : correct/len(predictions)
    }

training_args = TrainingArguments(output_dir='checkpoints/ambiguous_12layers/', 
                                    learning_rate=2e-3, 
                                    num_train_epochs=100,
                                    metric_for_best_model="original_loss",
                                    evaluation_strategy="epoch",
                                    logging_strategy="epoch",
                                    save_strategy="epoch",
                                    save_total_limit=1,
                                    per_device_train_batch_size=50,
                                    per_device_eval_batch_size=50,
                                    load_best_model_at_end=False)

trainer = Trainer(model = model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset={"original": eval, "ab": eval_ab, "a^2": eval_a, "b^2": eval_b},
                  compute_metrics=compute_metrics)

trainer.train()