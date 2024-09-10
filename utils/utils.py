import torch
import os
import random

def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name

# Ensure deterministic behavior
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False