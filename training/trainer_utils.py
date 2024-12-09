import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import List, Any, Dict, Optional

import wandb

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
            scheduler_params['T_max'] = 4 * args.epochs# // 2 or *2
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


def plot_shape_functions_during_training(X, phase_gams_out, epoch, num_features=10, num_outputs=4, vis_lat_features=False):
    """
    Plots the learned shape functions for each feature - during training.

    Parameters:
    - X (torch.Tensor): Input feature values, used for the x-axis.
    - phase_gams_out (torch.Tensor | dict): The learned shape functions.
    - epoch (int): For the plot title.
    - num_features (int): Number of features.
    - num_outputs (int): Number of outputs.
    - vis_lat_features (bool): Whether to visualize latent features.
    """
    # Ensure `X` is on the CPU and converted to NumPy for plotting
    if isinstance(phase_gams_out, torch.Tensor):
        phase_gams_out = phase_gams_out.detach().cpu().numpy()
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    else:
        X = X.cpu().numpy()

    # Determine global min and max for consistent y-axis limits
    global_min = float('inf')
    global_max = float('-inf')

    if isinstance(phase_gams_out, dict):
        for key, func in phase_gams_out.items():
            global_min = min(global_min, func.detach().cpu().numpy().min())
            global_max = max(global_max, func.detach().cpu().numpy().max())
    else:
        global_min = min(global_min, phase_gams_out.min())
        global_max = max(global_max, phase_gams_out.max())

    # Apply a 10% padding for better visualization
    y_min = float(global_min - 0.1 * abs(global_min))
    y_max = float(global_max + 0.1 * abs(global_max))

    # Create a figure with a subplot for each feature
    fig, axes = plt.subplots(num_features, num_outputs, figsize=(15, num_features * 3))

    for feature_idx in range(num_features):
        for output_idx in range(num_outputs):
            # Select the subplot for this feature and output
            ax = axes[feature_idx, output_idx] if num_features > 1 else axes[output_idx]

            # Retrieve learned and true shape functions
            if isinstance(phase_gams_out, dict):
                learned_shape_func = phase_gams_out[f'f_{output_idx}_{feature_idx}'].detach().cpu().numpy()
            else:
                learned_shape_func = phase_gams_out[:, output_idx, feature_idx]

            # Plot the learned shape function for the current feature and output
            ax.scatter(X[:, feature_idx], learned_shape_func, s=10, color='blue', alpha=0.6, label="Learned")

            # Set consistent y-limits across all subplots
            ax.set_ylim([y_min, y_max])

            # Set a title and labels
            ax.set_title(f'Feature {feature_idx} to Output {output_idx} - Epoch {epoch}')
            ax.set_xlabel("Input Values")
            ax.set_ylabel("Shape Function Output")
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if vis_lat_features:
        fig_name = f'Shape functions for phase1 Epoch {epoch}'
    else:
        fig_name = f'Shape functions for phase2 Epoch {epoch}'

    plt.savefig(f"training/plots/{fig_name}.png")

    # # Log the plot to W&B
    # if wandb.run is not None:
    #     wandb.log({f"Shape Functions/{fig_name}": wandb.Image(f"{fig_name}.png")})
        
    plt.close()
    return


def plot_data_histograms_during_training(values, values_name, epoch, nbins=50, save_path="data_processing/plots/"):
    """
    Plots histograms for each feature in the dataset and saves them for later exploration (raw features, concepts and targets).

    Parameters:
    -----------
    values : torch.Tensor
        The input tensor containing feature values.
    
    values_name : str
        The input type for the function. options: Input, Concept, Target

    nbins : int, optional (default=50)
        Number of bins for the histograms.

    model_predict : bool
        If True - load the model and get the results of the trained model
    
    save_path : str, optional (default='plots/')
        The path where the plots will be saved.
    """

    # Convert values to a pandas DataFrame for easier handling
    num_features = values.shape[1]
    df = pd.DataFrame(values.detach().cpu().numpy(), columns=[f'feature_{i}' for i in range(num_features)])

    fig = go.Figure()

    for i in range(num_features):
        fig.add_trace(
            go.Histogram(x=df[f'feature_{i}'], name=f'{values_name} {i}', opacity=0.75, nbinsx=nbins)
        )

    fig_title = f"predicted_{values_name}s_histogram_Epoch_{epoch}"

    # Update layout
    fig.update_layout(
        title=fig_title,
        xaxis_title='Value',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay histograms
        bargap=0.2,  # Gap between bars
        showlegend=True
    )

    # Save plot
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_filename = os.path.join(save_path, f"{fig_title}.png")
    fig.write_image(plot_filename)

    # if wandb.run is not None:
    #     wandb.log({f"data/{fig_title}": fig})
    return fig