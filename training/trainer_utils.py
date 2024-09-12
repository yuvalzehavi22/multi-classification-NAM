from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import List, Any, Dict, Optional


def l2_penalty(params, l2_lambda =0.):
    l2_penalty_val = l2_lambda * (params ** 2).sum() / params.shape[1]
    #print(l2_penalty_val)
    return l2_penalty_val

def l1_penalty(params, l1_lambda):
    l1_norm =  torch.stack([torch.linalg.norm(p, 1) for p in params], dim=0).sum()
    return l1_lambda*l1_norm

def penalized_binary_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    return F.binary_cross_entropy_with_logits(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

def penalized_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    return F.cross_entropy(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

def penalized_mse(logits, truth, fnn_out, feature_penalty=0.):
    return F.mse_loss(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

def penalized_mse(logits, truth):
    loss = F.mse_loss(logits.view(-1), truth.view(-1))
    #print(loss)
    return loss


def visualize_loss(train_loss_history,val_loss_history): 
    
    train_loss_history_np = [loss.detach().cpu().numpy() for loss in train_loss_history]
    
    if len(val_loss_history) > 0:
        val_loss_history_np = [loss.detach().cpu().numpy() for loss in val_loss_history]

    plt.figure(figsize=(12, 6))

    plt.plot(train_loss_history_np, color='r', label='Training')
    if len(val_loss_history) > 0:
        plt.plot(val_loss_history_np, color='r', label='Validation')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Loss")

    plt.legend()
    plt.show()
    return

def save_epoch_logs(epoch_logs: Dict, 
                    loss: float, 
                    score: Optional[Dict], 
                    stage: str
                    ):
    """
    Function to improve readability and avoid code repetition in the
    training/validation loop within the Trainer's fit method

    Parameters
    ----------
    epoch_logs: Dict
        Dict containing the epoch logs
    loss: float
        loss value
    score: Dict
        Dictionary where the keys are the metric names and the values are the
        corresponding values
    stage: str
        one of 'train' or 'val'
    """
    epoch_logs["_".join([stage, "loss"])] = loss
    if score is not None:
        for k, v in score.items():
            log_k = "_".join([stage, k])
            epoch_logs[log_k] = v
    return epoch_logs