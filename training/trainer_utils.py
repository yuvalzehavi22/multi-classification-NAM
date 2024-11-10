from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import List, Any, Dict, Optional

from training.monotonic_constraints import MonotonicityEnforcer
from utils.model_parser import *


def l2_penalty(params, l2_lambda =0.):
    l2_penalty_val = l2_lambda * (params ** 2).sum() / params.shape[1]
    #print(l2_penalty_val)
    return l2_penalty_val

def l1_penalty(params, l1_lambda = 0.):
    l1_norm = sum(p.abs().sum() for p in params)
    #normelaize
    total_params = sum(p.numel() for p in params)

    # # Monitor sparsity level
    # zero_params = sum((p == 0).sum().item() for p in params)
    # sparsity = zero_params / total_params * 100
    # print(f"Sparsity: {sparsity:.2f}% of parameters are zero")

    l1_penalty_val = l1_lambda*(l1_norm/total_params)
    return l1_penalty_val

def monotonic_penalty(input_data, output_data, mono_lambda = 0.):
    # Create an instance of MonotonicityEnforcer
    monotonicity_enforcer = MonotonicityEnforcer(input_data, output_data)
    # Compute the monotonicity penalty
    monotonicity_penalty = monotonicity_enforcer.compute_penalty()
    return monotonicity_penalty*mono_lambda

def visualize_loss(train_loss_history, val_loss_history): 
    
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


def set_lr_scheduler_params(args, lr_scheduler_type):
        """
        Setup lr_scheduler_params 
            
        Returns:
        --------
        scheduler_params : dict
            Parameters for the specific lr_scheduler
        """
        scheduler_params = {}

        if lr_scheduler_type == 'StepLR':
            scheduler_params['step_size'] = args.StepLR_step_size
            scheduler_params['gamma'] = args.StepLR_gamma
        
        elif lr_scheduler_type == 'CosineAnnealingLR':
            scheduler_params['T_max'] = args.epochs# // 2
            scheduler_params['eta_min'] = 0
            scheduler_params['last_epoch'] = -1

        elif lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler_params['mode'] = args.rop_mode
            scheduler_params['factor'] = args.rop_factor
            scheduler_params['patience'] = args.rop_patience
            scheduler_params['threshold'] = args.rop_threshold
            scheduler_params['min_lr'] = args.rop_min_lr

        elif lr_scheduler_type == 'CyclicLR':
            scheduler_params['base_lr'] = args.base_lr
            scheduler_params['max_lr'] = args.max_lr
            scheduler_params['step_size_up'] = args.step_size_up

        elif lr_scheduler_type == 'OneCycleLR':
            scheduler_params['max_lr'] = args.max_lr
            scheduler_params['total_steps'] = args.total_steps
            scheduler_params['epochs'] = args.epochs/100
            scheduler_params['steps_per_epoch'] = args.num_exp/args.batch_size
            scheduler_params['pct_start'] = args.pct_start
            scheduler_params['anneal_strategy'] = args.anneal_strategy

        elif lr_scheduler_type == 'NoScheduler':
            scheduler_params = None

        return scheduler_params

def get_param_groups(model, args):
    """
    Function to create group of the parameters for better visualization of the gradients
    """
    layers_name = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layers_name.append(name)

    # Initialize groups
    NAM_features_bias = []
    NAM_features_blocks = []
    NAM_output_bias = []
    NAM_output_blocks = []

    # Iterate over layer names and classify them
    for layer in layers_name:
        if layer == 'NAM_features.bias':
            NAM_features_bias.append(layer)
        elif 'NAM_features.feature_nns' in layer:
            block_idx = int(layer.split('.')[2])  # Extract the feature_nns block index
            if args.featureNN_arch_phase1 == 'parallel_single_output':
                sub_block_idx = int(layer.split('.')[4])  # Extract the sub-block index
                while len(NAM_features_blocks) <= block_idx:
                    NAM_features_blocks.append([])  # Add sublist for each output block
                if len(NAM_features_blocks[block_idx]) <= sub_block_idx:
                    NAM_features_blocks[block_idx].append([])  # Add sub-sublist for each output sub-block
                NAM_features_blocks[block_idx][sub_block_idx].append(layer)
            else:
                if len(NAM_features_blocks) <= block_idx:
                    NAM_features_blocks.append([])  # Add sublist for each feature block
                NAM_features_blocks[block_idx].append(layer)
        elif layer == 'NAM_output.bias':
            NAM_output_bias.append(layer)
        elif 'NAM_output.feature_nns' in layer:
            block_idx = int(layer.split('.')[2])  # Extract the output feature_nns block index
            if args.featureNN_arch_phase2 == 'parallel_single_output':
                sub_block_idx = int(layer.split('.')[4])  # Extract the sub-block index
                while len(NAM_output_blocks) <= block_idx:
                    NAM_output_blocks.append([])  # Add sublist for each output block
                if len(NAM_output_blocks[block_idx]) <= sub_block_idx:
                    NAM_output_blocks[block_idx].append([])  # Add sub-sublist for each output sub-block
                NAM_output_blocks[block_idx][sub_block_idx].append(layer)
            else:
                if len(NAM_output_blocks) <= block_idx:
                    NAM_output_blocks.append([])  # Add sublist for each feature block
                NAM_output_blocks[block_idx].append(layer)
    
    # Final Groups
    if args.hierarch_net:
        groups = {
            "NAM_features_bias": [NAM_features_bias],
            "NAM_features_blocks": NAM_features_blocks,
            "NAM_output_bias": [NAM_output_bias],
            "NAM_output_blocks": NAM_output_blocks
        }
    else:
        groups = {
            "NAM_features_bias": [NAM_features_bias],
            "NAM_features_blocks": NAM_features_blocks,
        }

    all_groups = {}
    for group_name, group_content in groups.items():
        count = 0
        for content in group_content:
            
            if len(content) == 2:
                for i in range(len(content)):
                    all_groups[f'{group_name}_{count}_for_outpot_{i}'] = content[i]
            else:
                all_groups[f'{group_name}_{count}'] = content
            count += 1 
    return all_groups