import random
import numpy as np
import torch

def set_seed(state, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state.seed = seed
    return state