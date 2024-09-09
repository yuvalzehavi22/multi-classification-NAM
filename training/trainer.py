import glob
import os
import time
from collections import OrderedDict
from os.path import join as pjoin, exists as pexists
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any

#from utils import define_device
from training.trainer_utils import l1_penalty, l2_penalty, define_device

class Trainer:
    """
    Trainer class for handling training and validation.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to be trained.
    
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    
    criterion : function
        Loss function to be used during training.
    
    lr_scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler for training.
    
    l1_lambda : float, optional
        Strength of L1 regularization (default 0.0).
    
    l2_lambda : float, optional
        Strength of L2 regularization (default 0.0).
    """
    
    def __init__(self, 
                 model, 
                 optimizer: Any = torch.optim.Adam, 
                 loss_function: Any =None, 
                 lr_scheduler: Any =None, 
                 device_name: str = "auto",
                 l1_lambda_phase1: int = 0.,
                 l1_lambda_phase2: int = 0.,
                 l2_lambda_phase1: int = 0.,
                 l2_lambda_phase2: int = 0.,
                 **kwargs):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Defining device
        self.device_name = device_name
        self.device = torch.device(define_device(self.device_name))

        # Regularization parameters
        self.l1_lambda_phase1 = l1_lambda_phase1
        self.l1_lambda_phase2 = l1_lambda_phase2
        self.l2_lambda_phase1 = l2_lambda_phase1
        self.l2_lambda_phase2 = l2_lambda_phase2

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
        
    def train(self, loader, config):
        """
        Runs the training process.
        
        Parameters:
        -----------
        loader : DataLoader
            DataLoader for the training set.
        
        config : dict
            Configuration parameters, including the number of epochs.
        """
        loss_history = []
        for epoch in tqdm(range(config['epochs'])):
            epoch_loss = self.train_epoch(loader)
            loss_history.append(epoch_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Total Loss: {epoch_loss:.5f}")
            
            if self.lr_scheduler:
                self.lr_scheduler.step()

        return loss_history

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
            logits, phase1_gams_out, phase2_gams_out = self.model(X)
        else:
            logits, phase1_gams_out = self.model(X)
            phase2_gams_out = None

        loss = self.criterion(logits.view(-1), y.view(-1))

        # Add L1 and L2 regularization for phase 1
        loss += l1_penalty(phase1_gams_out, self.l1_lambda_phase1)
        loss += l2_penalty(phase1_gams_out, self.l2_lambda_phase1)

        # Add L1 and L2 regularization for phase 2 if applicable
        if phase2_gams_out is not None:
            loss += l1_penalty(phase2_gams_out, self.l1_lambda_phase2)
            loss += l2_penalty(phase2_gams_out, self.l2_lambda_phase2)

        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()
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
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits, fnns_out = self.model(X)
                val_loss += self.criterion(logits, y, fnns_out)
        
        return val_loss / len(val_loader)



# class Trainer(nn.Module):

#     def __init__(self,
#                 model, 
#                 experiment_name=None, 
#                 warm_start=False,
#                 Optimizer=torch.optim.Adam, 
#                 optimizer_params={},
#                 lr=0.01, 
#                 lr_warmup_steps=-1, 
#                 l1_lambda_phase1 = 0.,
#                 l1_lambda_phase2 = 0.,
#                 l2_lambda_phase1 = 0.,
#                 l2_lambda_phase2 = 0.,
#                 verbose=False,
#                 n_last_checkpoints=5, 
#                 step_callbacks=[],
#                 **kwargs
#                 ):
          
#         """
#         Args:
#             model (torch.nn.Module): the model.
#             experiment_name: a path where all logs and checkpoints are saved.
#             warm_start: when set to True, loads the last checkpoint.
#             Optimizer: function(parameters) -> optimizer. Default: torch.optim.Adam.
#             optimizer_params: parameter when intializing optimizer. Usage:
#                 Optimizer(**optimizer_params).
#             verbose: when set to True, produces logging information.
#             n_last_checkpoints: the last few checkpoints to do model averaging.
#             step_callbacks: function(step). Will be called after each optimization step.
#         """
#         super().__init__()
        
#         self.model = model
#         self.verbose = verbose
#         self.lr = lr
#         self.lr_warmup_steps = lr_warmup_steps

#         # Regularization parameters
#         self.l1_lambda_phase1 = l1_lambda_phase1,
#         self.l1_lambda_phase2 = l1_lambda_phase2,
#         self.l2_lambda_phase1 = l2_lambda_phase1,
#         self.l2_lambda_phase2 = l2_lambda_phase2,
        
#         params = [p for p in self.model.parameters() if p.requires_grad]
        
#         self.opt = Optimizer(params, lr=lr, **optimizer_params)
#         self.step = 0
#         self.n_last_checkpoints = n_last_checkpoints
#         self.step_callbacks = step_callbacks
        
#         if self.model.task_type == 'binary_classification':
#             self.criterion = F.binary_cross_entropy_with_logits()
#         elif self.model.task_type == 'multi_classification':
#             self.criterion = F.cross_entropy()
#         elif self.model.task_type == 'regression':
#             self.criterion = F.mse_loss()
#         else:
#             raise NotImplementedError()
        
#         if experiment_name is None:
#             experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
#             if self.verbose:
#                 print('using automatic experiment name: ' + experiment_name)
        
#         self.experiment_path = pjoin('logs/', experiment_name)
        
#         if warm_start:
#             self.load_checkpoint()


#     def fit(
#         self,
#         X_train,
#         y_train,
#         eval_set=None,
#         eval_name=None,
#         eval_metric=None,
#         loss_fn=None,
#         weights=0,
#         max_epochs=100,
#         patience=10,
#         batch_size=1024,
#         virtual_batch_size=128,
#         num_workers=0,
#         drop_last=True,
#         callbacks=None,
#         pin_memory=True,
#         from_unsupervised=None,
#         warm_start=False,
#         augmentations=None,
#         compute_importance=True
#     ):
#         """Train a neural network stored in self.network
#         Using train_dataloader for training data and
#         valid_dataloader for validation.

#         Parameters
#         ----------
#         X_train : np.ndarray
#             Train set
#         y_train : np.array
#             Train targets
#         eval_set : list of tuple
#             List of eval tuple set (X, y).
#             The last one is used for early stopping
#         eval_name : list of str
#             List of eval set names.
#         eval_metric : list of str
#             List of evaluation metrics.
#             The last metric is used for early stopping.
#         loss_fn : callable or None
#             a PyTorch loss function
#         weights : bool or dictionnary
#             0 for no balancing
#             1 for automated balancing
#             dict for custom weights per class
#         max_epochs : int
#             Maximum number of epochs during training
#         patience : int
#             Number of consecutive non improving epoch before early stopping
#         batch_size : int
#             Training batch size
#         virtual_batch_size : int
#             Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
#         num_workers : int
#             Number of workers used in torch.utils.data.DataLoader
#         drop_last : bool
#             Whether to drop last batch during training
#         warm_start: bool
#             If True, current model parameters are used to start training
#         """
#         # update model name

#         self.max_epochs = max_epochs
#         self.patience = patience
#         self.batch_size = batch_size
#         self.virtual_batch_size = virtual_batch_size
#         self.num_workers = num_workers
#         self.drop_last = drop_last
#         self.input_dim = X_train.shape[1]
#         self._stop_training = False
#         self.pin_memory = pin_memory and (self.device.type != "cpu")
        
#         eval_set = eval_set if eval_set else []

#         if loss_fn is None:
#             self.loss_fn = self._default_loss
#         else:
#             self.loss_fn = loss_fn

#         check_input(X_train)
#         check_warm_start(warm_start, from_unsupervised)

#         self.update_fit_params(
#             X_train,
#             y_train,
#             eval_set,
#             weights,
#         )

#         # Validate and reformat eval set depending on training data
#         eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

#         train_dataloader, valid_dataloaders = self._construct_loaders(
#             X_train, y_train, eval_set
#         )

#         if from_unsupervised is not None:
#             # Update parameters to match self pretraining
#             self.__update__(**from_unsupervised.get_params())

#         if not hasattr(self, "network") or not warm_start:
#             # model has never been fitted before of warm_start is False
#             self._set_network()
#         self._update_network_params()
#         self._set_metrics(eval_metric, eval_names)
#         self._set_optimizer()
#         self._set_callbacks(callbacks)

#         if from_unsupervised is not None:
#             self.load_weights_from_unsupervised(from_unsupervised)
#             warnings.warn("Loading weights from unsupervised pretraining")
#         # Call method on_train_begin for all callbacks
#         self._callback_container.on_train_begin()

#         # Training loop over epochs
#         for epoch_idx in range(self.max_epochs):

#             # Call method on_epoch_begin for all callbacks
#             self._callback_container.on_epoch_begin(epoch_idx)

#             self._train_epoch(train_dataloader)

#             # Apply predict epoch to all eval sets
#             for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
#                 self._predict_epoch(eval_name, valid_dataloader)

#             # Call method on_epoch_end for all callbacks
#             self._callback_container.on_epoch_end(
#                 epoch_idx, logs=self.history.epoch_metrics
#             )

#             if self._stop_training:
#                 break

#         # Call method on_train_end for all callbacks
#         self._callback_container.on_train_end()
#         self.network.eval()

#         if self.compute_importance:
#             # compute feature importance once the best model is defined
#             self.feature_importances_ = self._compute_feature_importances(X_train)

#     def predict(self, X):
#         """
#         Make predictions on a batch (valid)

#         Parameters
#         ----------
#         X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
#             Input data

#         Returns
#         -------
#         predictions : np.array
#             Predictions of the regression problem
#         """
#         self.network.eval()

#         if scipy.sparse.issparse(X):
#             dataloader = DataLoader(
#                 SparsePredictDataset(X),
#                 batch_size=self.batch_size,
#                 shuffle=False,
#             )
#         else:
#             dataloader = DataLoader(
#                 PredictDataset(X),
#                 batch_size=self.batch_size,
#                 shuffle=False,
#             )

#         results = []
#         for batch_nb, data in enumerate(dataloader):
#             data = data.to(self.device).float()
#             output, M_loss = self.network(data)
#             predictions = output.cpu().detach().numpy()
#             results.append(predictions)
#         res = np.vstack(results)
#         return self.predict_func(res)

