import numpy as np
import os
import torch
import torch.nn


def set_seed(seed: int) -> torch.Generator:
    # Try to make this implementation as deterministic as possible.
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
