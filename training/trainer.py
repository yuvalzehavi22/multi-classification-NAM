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
import torch.nn.functional as F
from typing import List, Any, Dict
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import wandb
import plotly.graph_objects as go

#from utils import define_device
from training.trainer_utils import l1_penalty, l2_penalty, monotonic_penalty
from utils.utils import define_device
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
                 monotonicity_lambda: float = 0.,
                 eval_every: int = 1,
                 early_stop_delta: float =0.001,
                 early_stop_patience: int =10,
                 clip_value: int =None,
                 device_name: str ="auto"):
        
        self.model = model
        self.optimizer = self._set_optimizer(optimizer, learning_rate, weight_decay)
        self.lr_scheduler = self._set_lr_scheduler(self.optimizer, lr_scheduler, scheduler_params)
        self.eval_metric = eval_metric
        self.epochs = epochs
        self.batch_size = batch_size
        # Regularization parameters
        self.l1_lambda_phase1 = l1_lambda_phase1
        self.l2_lambda_phase1 = l2_lambda_phase1
        self.l1_lambda_phase2 = l1_lambda_phase2
        self.l2_lambda_phase2 = l2_lambda_phase2
        self.monotonicity_lambda = monotonicity_lambda
        self.eval_every = eval_every
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience
        self.clip_value = clip_value

        # Defining device
        self.device = torch.device(define_device(device_name))

        # Best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

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
        
    def train(self, args, loader, all_param_groups=None, val_loader=None):
        """
        Runs the training process.
        
        Parameters:
        -----------
        loader : DataLoader
            DataLoader for the training set.
        
        val_loader : DataLoader, optional
            DataLoader for the validation set (for early stopping).
        """
        # # Initialize W&B run and log the parameters
        # wandb.init(project="Hirarchial GAMs", config=args)

        train_loss_history = []
        val_loss_history = []

        # Initialize a dictionary to store gradient norms
        grad_norms = {}

        for epoch in tqdm(range(self.epochs)):
            epoch_loss = self.train_epoch(loader)
            train_loss_history.append(epoch_loss)  

            # Track and log gradients
            self._track_gradients(grad_norms, epoch)
            #self._log_gradients_wandb(grad_norms, epoch)
           
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Early Stopping Check
            if val_loader:
                val_loss, rmse = self.validate(val_loader)
                val_loss_history.append(val_loss)
                
                if epoch % self.eval_every == 0:
                    print(f"Epoch {epoch} | Total Loss: {epoch_loss:.5f} | Validation Loss: {val_loss:.5f}")
                    # Log metrics to W&B
                    wandb.log({"Epoch": epoch, "Training loss": epoch_loss, "Validation Loss": val_loss, "Validation RMSE": rmse})

                if val_loss < self.best_val_loss - self.early_stop_delta:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    self.save_model("best_model.pt")
                    wandb.save(f"model_epoch_{epoch}.pt")
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stop_patience:
                        print("Early stopping triggered!")
                        break

                # # Learning rate scheduler step (if using ReduceLROnPlateau, pass validation loss)
                # if self.lr_scheduler:
                #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                #         self.lr_scheduler.step(val_loss)
                #     else:
                #         self.lr_scheduler.step()
            
            else:
                if epoch % self.eval_every == 0:
                    print(f"Epoch {epoch} | Total Loss: {epoch_loss:.5f}")
                    # Log metrics to W&B
                    wandb.log({"Epoch": epoch, "Training loss": epoch_loss})
        
        if all_param_groups:
            # Plot the gradients at the end of training and log the plot to W&B
            print('Plot the gradients at the end of training...')
            self._plot_gradients(grad_norms, all_param_groups)
            #wandb.log({"Gradient Norms Plot": wandb.Image("gradient_norms.png")})

        return train_loss_history, val_loss_history
    

    def train_epoch(self, loader):
        """
        Performs one epoch of training.
        
        Parameters:
        -----------
        loader : DataLoader
            DataLoader for the training set.
        
        Returns:
        --------
        avg_loss : float
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        
        for X, y in loader:
            loss = self.train_batch(X, y)
            epoch_loss += loss

        return epoch_loss / len(loader)

    def train_batch(self, X, y):
        """
        Performs a single batch training step.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input features.
        
        y : torch.Tensor
            Target values.
        
        Returns:
        --------
        loss : float
            Computed loss for the batch.
        """
        X, y = X.to(self.device), y.to(self.device)
        
        # Forward pass
        if self.model.hierarch_net:
            logits, latent_features, phase1_gams_out, phase2_gams_out = self.model(X)
        else:
            logits, phase1_gams_out = self.model(X)
            phase2_gams_out = None
        
        loss = self.criterion(logits.view(-1), y.view(-1))

        # Add L1 and L2 regularization for phase 1
        #loss += l1_penalty(self.model, self.l1_lambda_phase1)
        loss += l1_penalty(phase1_gams_out, self.l1_lambda_phase1)
        loss += l2_penalty(phase1_gams_out, self.l2_lambda_phase1)

        # Add L1 and L2 regularization for phase 2 if applicable
        if phase2_gams_out is not None:
            loss += l1_penalty(phase2_gams_out, self.l1_lambda_phase2)
            loss += l2_penalty(phase2_gams_out, self.l2_lambda_phase2)

            # Add Monotonicity Penalty
            loss += monotonic_penalty(latent_features, logits, self.monotonicity_lambda)

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_value:
            clip_grad_norm_(self.model.parameters(), self.clip_value)
        
        self.optimizer.step()

        return loss.item()

    def validate(self, val_loader):
        """
        Runs validation.
        
        Parameters:
        -----------
        val_loader : DataLoader
            DataLoader for the validation set.
        
        Returns:
        --------
        val_loss : float
            Validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        rmse = 0.0

        list_y_true = []
        list_y_pred = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                if self.model.hierarch_net:
                    logits, _, _, _ = self.model(X)
                else:
                    logits, _ = self.model(X)

                # Calculate loss
                val_loss += self.criterion(logits.view(-1), y.view(-1))

                # Store predictions and ground truths for later metrics calculation
                list_y_true.append(y.cpu())
                list_y_pred.append(logits.cpu())

            # Concatenate all predictions and true labels
            y_true = torch.cat(list_y_true)
            y_pred = torch.cat(list_y_pred)

            # Calculate overall RMSE on the entire validation set
            rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

            # Average validation loss over the number of batches
            avg_val_loss = val_loss / len(val_loader)
        
        return avg_val_loss, rmse

    def _set_optimizer(self, name, lr, wd):
        """
        Setup optimizer
        
        Parameters:
        -----------
        name : Optimizer name
        lr : larning rate
        wd : Wigth decay
        
        Returns:
        --------
        optimizer function
        """
        if name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=lr,
                                 weight_decay=wd,
                                )
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                 lr=lr,
                                 weight_decay=wd,
                                )
        return optimizer
    

    def _set_lr_scheduler(self, optimizer, lr_scheduler_type, scheduler_params):
        """
        Setup lr_scheduler
        
        Parameters:
        -----------
        lr_scheduler_type : str
            Name of the lr_scheduler to use
        scheduler_params : dict
            Parameters for the specific lr_scheduler
        
        Returns:
        --------
        lr_scheduler function
        """
        if lr_scheduler_type == 'StepLR':
            #lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=scheduler_params.get('step_size', 10), 
                                                       gamma=scheduler_params.get('gamma', 0.1))

        elif lr_scheduler_type == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_params.get('mode', 'min'),
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
            wandb.log({f"Gradient Norms Plot: {group_name}": fig})

    def _log_gradients_wandb(self, grad_norms, epoch):
        """
        Logging gradients with W&B
        """
        for layer in grad_norms[epoch].keys():
            wandb.log({layer: grad_norms[epoch][layer]})

    def save_model(self, path):
        """
        Save the model state to a file.
        
        Parameters:
        -----------
        path : str
            File path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        #print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model state from a file.
        
        Parameters:
        -----------
        path : str
            File path to load the model from.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")