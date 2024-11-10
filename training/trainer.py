import glob
import os
import time
from collections import OrderedDict
from os.path import join as pjoin, exists as pexists
from matplotlib import pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
import torch.nn.functional as F
from typing import List, Any, Dict
from torch.nn.utils import clip_grad_norm_
import wandb
import plotly.graph_objects as go

#from utils import define_device
from training.trainer_utils import l1_penalty, l2_penalty, monotonic_penalty
from utils.utils import define_device, plot_data_histograms
from utils.model_parser import *


class Trainer:
    """
    Trainer class for handling training and validation.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to be trained.
    
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    
    lr_scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler for training.
    
    scheduler_params : dict
        Scheduler parameters to be passed to the learning rate scheduler.
    
    eval_metric : str
        Metric used for evaluation (accuracy, F1-score, etc.).
    
    epochs : int
        Number of training epochs.
    
    batch_size : int
        Batch size for training.
    
    learning_rate : float
        Initial learning rate for training.
    
    weight_decay : float
        Weight decay (L2 regularization).
    
    l1_lambda_phase1 : float
        Strength of L1 regularization for phase 1.
    
    l2_lambda_phase1 : float
        Strength of L2 regularization for phase 1.
    
    l1_lambda_phase2 : float
        Strength of L1 regularization for phase 2.
    
    l2_lambda_phase2 : float
        Strength of L2 regularization for phase 2.
    
    eval_every : int
        Evaluate the model every N epochs.
    
    early_stop_delta : float
        Minimum change to qualify as an improvement for early stopping.
    
    early_stop_patience : int
        Number of epochs with no improvement after which training will be stopped.
    """
    
    def __init__(self, 
                 model,
                 optimizer: str = 'Adam', 
                 loss_function: Any =None, 
                 lr_scheduler: Any =None, 
                 scheduler_params: Dict = None,
                 eval_metric: List = None, 
                 epochs: int =100, 
                 batch_size: int =32, 
                 learning_rate: float =0.001, 
                 weight_decay: float =0.0, 
                 l1_lambda_phase1: float = 0.,
                 l1_lambda_phase2: float = 0.,
                 l2_lambda_phase1: float = 0.,
                 l2_lambda_phase2: float = 0.,
                 monotonicity_lambda_phase1: float = 0.,
                 monotonicity_lambda_phase2: float = 0.,
                 eval_every: int = 1,
                 early_stop_delta: float =0.001,
                 early_stop_patience: int =10,
                 clip_value: int =None,
                 device_name: str ="auto"):
        
        self.model = model
        self.optimizer = self._set_optimizer(optimizer, learning_rate, weight_decay)
        self.lr_scheduler_type = lr_scheduler
        self.lr_scheduler = self._set_lr_scheduler(self.optimizer, lr_scheduler, scheduler_params)
        self.eval_metric = eval_metric
        self.epochs = epochs
        self.batch_size = batch_size
        # Regularization parameters
        self.l1_lambda_phase1 = l1_lambda_phase1
        self.l2_lambda_phase1 = l2_lambda_phase1
        self.l1_lambda_phase2 = l1_lambda_phase2
        self.l2_lambda_phase2 = l2_lambda_phase2
        self.monotonicity_lambda_phase1 = monotonicity_lambda_phase1
        self.monotonicity_lambda_phase2 = monotonicity_lambda_phase2
        self.eval_every = eval_every
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience
        self.clip_value = clip_value

        # Defining device
        self.device = torch.device(define_device(device_name))

        # Automatic loss function based on task type
        if loss_function is None:
            if self.model.task_type == 'binary_classification':
                self.criterion = F.binary_cross_entropy_with_logits
            elif self.model.task_type == 'multi_classification':
                self.criterion = F.cross_entropy
            elif self.model.task_type == 'regression':
                self.criterion = F.mse_loss
            else:
                raise NotImplementedError()
            
        # User-defined loss function
        else:
            self.criterion = loss_function
        
    def train(self, loader, all_param_groups=None, val_loader=None):
        """
        Runs the training process.
        """

        train_loss_history = []
        val_loss_history = []

        # Initialize variables for early stopping
        best = float('inf')
        early_stop_counter = 0

        # Initialize a dictionary to store gradient norms
        grad_norms = {}

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            epoch_loss, epoch_mae, epoch_rmse = 0.0, 0.0, 0.0

            # Training loop
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                if self.model.hierarch_net or self.model.learn_only_concepts:
                    logits, latent_features, phase1_gams_out, phase2_gams_out = self.model(X)
                else:
                    logits, phase1_gams_out = self.model(X)
                    latent_features, phase2_gams_out = None, None
                
                # Calculate loss
                loss = self.criterion(logits.view(-1), y.view(-1))

                # Add L1, L2 regularization and Monotonicity Penalty for phase 1
                #l1_penalty_phase1 = l1_penalty(phase1_gams_out, self.l1_lambda_phase1)
                l2_penalty_phase1 = l2_penalty(phase1_gams_out, self.l2_lambda_phase1)
                loss += l2_penalty_phase1
                #loss += l1_penalty_phase1 + l2_penalty_phase1 + mono_penalty_phase1

                params = [param for name, param in self.model.named_parameters() if 'multi_output_layer' in name]
                if len(params) > 0:
                    l1_penalty_phase1_arch = l1_penalty(params, self.l1_lambda_phase1)
                    loss += l1_penalty_phase1_arch

                # Add L1, L2 regularization and Monotonicity Penalty for phase 2 if applicable
                if self.model.hierarch_net and not self.model.learn_only_concepts:
                    l1_penalty_phase2 = l1_penalty(phase2_gams_out, self.l1_lambda_phase2)
                    l2_penalty_phase2 = l2_penalty(phase2_gams_out, self.l2_lambda_phase2)
                    mono_penalty_phase2 = monotonic_penalty(latent_features, logits, self.monotonicity_lambda_phase2)
                    loss += l1_penalty_phase2 + l2_penalty_phase2 + mono_penalty_phase2

                # Backward pass and optimization step
                loss.backward()

                if self.clip_value:
                    clip_grad_norm_(self.model.parameters(), self.clip_value)
                
                self.optimizer.step()

                # Calculate metrics without gradient tracking
                with torch.no_grad():
                    mae = l1_loss(logits.view(-1), y.view(-1))  # MAE calculation
                    rmse = torch.sqrt(torch.mean((logits.view(-1) - y.view(-1)) ** 2))  # RMSE calculation

                epoch_loss += loss.item()
                epoch_mae += mae.item()
                epoch_rmse += rmse.item()

            # Calculate the average of the metrics for the current epoch
            train_loss = epoch_loss / len(loader)
            train_mae = epoch_mae / len(loader)
            train_rmse = epoch_rmse / len(loader)

            train_loss_history.append(train_loss)  

            # Log losses and learning rate to W&B
            wandb.log({
                #"epoch": epoch + 1,
                "train/train_loss": train_loss,
                "train/train_mae": train_mae,
                "train/train_rmse": train_rmse,
                "train/learning_rate": self.lr_scheduler.get_last_lr()[0]  # Log current learning rate
            })

            # # Track and log gradients
            # self._track_gradients(grad_norms, epoch)

            if val_loader is not None:
                val_loss, val_rmse, val_mae = self.validate(val_loader)
                val_loss_history.append(val_loss)

                if self.lr_scheduler:
                    if self.lr_scheduler_type == 'ReduceLROnPlateau':  
                        # Adjust learning rate based on val_loss
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()
                
                if epoch % self.eval_every == 0:
                    print(f"Epoch {epoch} | Total Loss: {epoch_loss:.5f} | Validation Loss: {val_loss:.5f}")
                
                # Log losses and learning rate to W&B
                wandb.log({
                    #"epoch": epoch + 1,
                    "validation/val_loss": val_loss,
                    "validation/val_mae": val_mae,
                    "validation/val_rmse": val_rmse,
                })

                # Early Stopping Check
                if val_loss < best - self.early_stop_delta:
                    best = val_loss
                    early_stop_counter = 0
                    self.save_model("best_model.pt")
                else:
                    early_stop_counter += 1
                
                if early_stop_counter == self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            else:
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                if epoch % self.eval_every == 0:
                    print(f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Train MAE: {train_mae:.5f} | Train RMSE: {train_rmse:.5f}")

        if val_loader is None:
            self.save_model("best_model.pt")        
        
        # # Plot the gradients at the end of training and log the plot to W&B
        # if all_param_groups:
        #     print('Plot the gradients at the end of training...')
        #     self._plot_gradients(grad_norms, all_param_groups)

        return train_loss_history, val_loss_history

    def validate(self, val_loader):
        """
        Runs validation.
        """
        self.model.eval()
        val_loss, rmse, mae = 0.0, 0.0, 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                if self.model.hierarch_net:
                    logits, _, _, _ = self.model(X)
                else:
                    logits, _ = self.model(X)

                # Calculate loss
                val_loss += self.criterion(logits.view(-1), y.view(-1))

                # MAE and RMSE calculations
                mae += l1_loss(logits.view(-1), y.view(-1)).item()
                rmse += torch.sqrt(torch.mean((logits.view(-1) - y.view(-1)) ** 2)).item()

            # Average validation loss, MAE, and RMSE over the number of batches
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = mae / len(val_loader)
            avg_val_rmse = rmse / len(val_loader)

        return avg_val_loss, avg_val_rmse, avg_val_mae

    def _set_optimizer(self, optimizer_name, lr, wd):
        """
        Setup optimizer
        """
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=lr,
                                 weight_decay=wd,
                                )
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                 lr=lr,
                                 weight_decay=wd,
                                )
        return optimizer
    

    def _set_lr_scheduler(self, optimizer, lr_scheduler_type, scheduler_params):
        """
        Setup lr_scheduler
        """
        if lr_scheduler_type == 'StepLR':
            #lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=scheduler_params.get('step_size', 10), 
                                                       gamma=scheduler_params.get('gamma', 0.1))
            
        elif lr_scheduler_type == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                      T_max=scheduler_params.get('T_max', 20), 
                                                                      eta_min=scheduler_params.get('eta_min', 0), 
                                                                      last_epoch=scheduler_params.get('last_epoch', -1))

        elif lr_scheduler_type == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode=scheduler_params.get('mode', 'min'),
                                                                factor=scheduler_params.get('factor', 0.1),
                                                                patience=scheduler_params.get('patience', 10),
                                                                threshold=scheduler_params.get('threshold', 0.0001),
                                                                min_lr=scheduler_params.get('min_lr', 0))

        elif lr_scheduler_type == 'CyclicLR':
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                     base_lr=scheduler_params.get('base_lr', 0.001),
                                                     max_lr=scheduler_params.get('max_lr', 0.1),
                                                     step_size_up=scheduler_params.get('step_size_up', 2000))

        elif lr_scheduler_type == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=scheduler_params.get('max_lr', 0.01),
                                                       total_steps=scheduler_params.get('total_steps', None),
                                                       epochs=scheduler_params.get('epochs', 10),
                                                       steps_per_epoch=scheduler_params.get('steps_per_epoch', 100),
                                                       pct_start=scheduler_params.get('pct_start', 0.3),
                                                       anneal_strategy=scheduler_params.get('anneal_strategy', 'linear'))

        elif lr_scheduler_type == 'NoScheduler':
            lr_scheduler = None

        return lr_scheduler

    # Function to track and store gradient norms
    def _track_gradients(self, grad_norms, epoch):
        """
        track the magnitude of the gradients layer by layer during training
        """
        grad_norms[epoch] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None: #and name in layers_to_track 
               grad_norms[epoch][name] = param.grad.norm().item()
    
    # Function to plot gradient norms
    def _plot_gradients(self, grad_norms, all_groups):
        for group_name, group_content in all_groups.items():
            fig = go.Figure()
            for layer in group_content:
                if isinstance(layer, list):
                    # Check if 'layer' is a list - for 'parallel_single_output' arch
                    for sub_layer in layer:  
                        grad_values = [grad_norms[epoch][sub_layer] for epoch in grad_norms]
                        fig.add_trace(go.Scatter(x=list(grad_norms.keys()), y=grad_values, mode='lines', name=sub_layer))
                else:
                    grad_values = [grad_norms[epoch][layer] for epoch in grad_norms]
                    fig.add_trace(go.Scatter(x=list(grad_norms.keys()), y=grad_values, mode='lines', name=layer))

            fig.update_layout(
                title=f"Gradient Norms - {group_name}",
                xaxis_title="Epoch",
                yaxis_title="Gradient Norm",
                legend=dict(
                    x=1.05,  # Positioning the legend outside the figure
                    y=0.5,
                    traceorder="normal",
                    font=dict(size=10),
                    bordercolor="Black",
                    borderwidth=1
                )
            )

            # fig.show()
            # Log the figure to W&B
            wandb.log({f"gradients/Gradient Norms Plot: {group_name}": fig})

    def _log_gradients_wandb(self, grad_norms, epoch):
        """
        Logging gradients with W&B
        """
        for layer in grad_norms[epoch].keys():
            wandb.log({layer: grad_norms[epoch][layer]})

    def save_model(self, path):
        """
        Save the model state to a file.
        """
        
        torch.save(self.model.state_dict(), path)
        #print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model state from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def plot_pred_data_histograms(self, loader):
        """
        Plots histograms of the predicted output.
        using the "plot_data_histograms" function in the "utils/utils.py" file
        """
        self.model.eval()
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                if self.model.hierarch_net:
                    logits, latent_features, _, _ = self.model(X)
                    _ = plot_data_histograms(values=latent_features, values_name='Concept',nbins=80, model_predict=True, save_path="data_processing/plots/")
                    _ = plot_data_histograms(values=logits, values_name='Target',nbins=100, model_predict=True, save_path="data_processing/plots/")
                else:
                    logits, _ = self.model(X)
                    _ = plot_data_histograms(values=logits, values_name='Concept',nbins=80, model_predict=True, save_path="data_processing/plots/")


