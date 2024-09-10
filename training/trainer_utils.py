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

# def get_loss_function(task_type):
#     """
#     Selects the appropriate loss function based on the task type.
    
#     Parameters:
#     ----------
#     task_type : str
#         Type of the task: 'regression', 'binary_classification', 'multi_classification'.
        
#     Returns:
#     -------
#     loss_fn : function
#         The selected loss function.
#     """
#     if task_type == 'binary_classification':
#         return penalized_binary_cross_entropy
#     elif task_type == 'multi_classification':
#         return penalized_cross_entropy
#     elif task_type == 'regression':
#         return penalized_mse
#     else:
#         raise ValueError(f"Unknown task type: {task_type}")

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


# def set_loss_function(loss_fn: str, **kwargs):
#     r"""Function that returns the corresponding loss function given an alias

#     Parameters
#     ----------
#     loss_fn: str
#         Loss name or alias

#     Returns
#     -------
#     Object
#         loss function

#     Examples
#     --------
#     >>> from pytorch_widedeep.training._trainer_utils import alias_to_loss
#     >>> loss_fn = alias_to_loss(loss_fn="binary_logloss", weight=None)
#     """

#     if loss_fn in _LossAliases.get("binary"):
#         return nn.BCEWithLogitsLoss(pos_weight=kwargs["weight"])
#     elif loss_fn in _LossAliases.get("multiclass"):
#         return nn.CrossEntropyLoss(weight=kwargs["weight"])
#     elif loss_fn in _LossAliases.get("regression"):
        
#         return MSELoss()
#     elif loss_fn in _LossAliases.get("mean_absolute_error"):
#         return L1Loss()
#     elif loss_fn in _LossAliases.get("mean_squared_log_error"):
#         return MSLELoss()
#     elif loss_fn in _LossAliases.get("root_mean_squared_error"):
#         return RMSELoss()
#     elif loss_fn in _LossAliases.get("root_mean_squared_log_error"):
#         return RMSLELoss()
    
#     # elif "focal_loss" in loss_fn:
#     #     return FocalLoss(**kwargs)
#     else:
#         raise ValueError(
#             "objective or loss function is not supported."
#         )


# def penalized_mse(logits, truth, fnn_out, feature_penalty=0.0):
#     feat_loss = feature_loss(fnn_out, feature_penalty)
#     mse_loss = F.mse_loss(logits.view(-1), truth.view(-1))
#     loss = mse_loss+feat_loss
#     return loss

    
# # Regularization
# def feature_loss(fnn_out, lambda_=0.):
#     return lambda_ * (fnn_out ** 2).sum() / fnn_out.shape[1]

# def l1_penalty(params, l1_lambda):
#     l1_norm =  torch.stack([torch.linalg.norm(p, 1) for p in params], dim=0).sum()
#     return l1_lambda*l1_norm

# def l2_penalty(params, l1_lambda):
#     l2_norm =  torch.stack([torch.linalg.norm(p, 2) for p in params], dim=0).sum()
#     return l1_lambda*l2_norm