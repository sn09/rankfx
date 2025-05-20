"""Module with utils for training process."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all kinds of seeds.

    Args:
        seed: seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
