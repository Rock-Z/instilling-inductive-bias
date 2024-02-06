import numpy as np
import torch
from torch.utils.data import Dataset

class Numbers(Dataset):
    """
    Dataset & Tokenizer in one for generation output from `generate_samples`. Outputs `input_ids` and `labels`
    for self-supervised training of HuggingFace GPT-2
    """
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

def generate_samples(n_samples : int, max: int, p: int, ambiguous : bool = False, computation: str = "a^2 + ab"):
    """
    Generate training samples for a given calculation
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
    assert computation in ["a + ab", "a^2 + ab", "2a^2", "2b^2", "a^2 + ab + b^2", "a^3 + ab", "a^2 + b^2", "a^2 + ab + b^2 + a", "a^3 + ab + b", "ab", "a^2", "a + b", "a^2 - b^2"]
    
    if computation == "a^2 + ab":
        X[:, 3] = np.mod(np.square(X[:, 0]) + X[:, 0] * X[:, 1], p)
    elif computation == "a + ab":
        X[:, 3] = np.mod(X[:, 0] + X[:, 0] * X[:, 1], p)
    elif computation == "2a^2":
        X[:, 3] = np.mod(2 * np.square(X[:, 0]), p)
    elif computation == "2b^2":
        X[:, 3] = np.mod(2 * np.square(X[:, 1]), p)
    elif computation == "a^2 + ab + b^2":
        X[:, 3] = np.mod(np.square(X[:, 0]) + X[:, 0] * X[:, 1] + np.square(X[:, 1]), p)
    elif computation == "a^3 + ab":
        X[:, 3] = np.mod(np.power(X[:, 0], 3) + X[:, 0] * X[:, 1], p)
    elif computation == "a^2 + b^2":
        X[:, 3] = np.mod(np.square(X[:, 0]) + np.square(X[:, 1]), p)
    elif computation == "a^2 + ab + b^2 + a":
        X[:, 3] = np.mod(np.square(X[:, 0]) + X[:, 0] * X[:, 1] + np.square(X[:, 1]) + X[:, 0], p)
    elif computation == "a^3 + ab + b":
        X[:, 3] = np.mod(np.power(X[:, 0], 3) + X[:, 0] * X[:, 1] + X[:, 1], p)
    elif computation == "a^2":
        X[:, 3] = np.mod(np.square(X[:, 0]), p)
    elif computation == "ab":
        X[:, 3] = np.mod(X[:, 0] * X[:, 1], p)
    elif computation == "a + b":
        X[:, 3] = np.mod(X[:, 0] + X[:, 1], p)
    elif computation == "a^2 - b^2":
        X[:, 3] = np.mod(np.square(X[:, 0]) - np.square(X[:, 1]), p)
    
    return X

def set_random_seed(seed : int):
    """Fix torch and numpy seed for run reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)    
    
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