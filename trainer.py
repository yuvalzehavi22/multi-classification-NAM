import glob
import os
import time
from collections import OrderedDict
from copy import deepcopy
from os.path import join as pjoin, exists as pexists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class Trainer(nn.Module):

    def __init__(self,
                model, 
                experiment_name=None, 
                warm_start=False,
                Optimizer=torch.optim.Adam, 
                optimizer_params={},
                lr=0.01, 
                lr_warmup_steps=-1, 
                verbose=False,
                n_last_checkpoints=5, 
                step_callbacks=[],
                problem='classification', 
                **kwargs
                ):
          
        """
        Args:
            model (torch.nn.Module): the model.
            experiment_name: a path where all logs and checkpoints are saved.
            warm_start: when set to True, loads the last checkpoint.
            Optimizer: function(parameters) -> optimizer. Default: torch.optim.Adam.
            optimizer_params: parameter when intializing optimizer. Usage:
                Optimizer(**optimizer_params).
            verbose: when set to True, produces logging information.
            n_last_checkpoints: the last few checkpoints to do model averaging.
            step_callbacks: function(step). Will be called after each optimization step.
            problem: problem type. Chosen from ['classification', 'regression'].
        """
        super().__init__()
        
        self.model = model
        self.verbose = verbose
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.opt = Optimizer(params, lr=lr, **optimizer_params)
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints
        self.step_callbacks = step_callbacks
        self.problem = problem
        
        if problem == 'classification':
            self.loss_function = (
				lambda x, y: F.binary_cross_entropy_with_logits(x, y.float()) if x.ndim == 1
					else F.cross_entropy(x, y)
			)
            
        elif problem == 'regression':
            self.loss_function = (lambda y1, y2: F.mse_loss(y1.float(), y2.float()))
        
        else:
            raise NotImplementedError()
        
        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)
        
        self.experiment_path = pjoin('logs/', experiment_name)
        
        if warm_start:
            self.load_checkpoint()


    


    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
            """This is for evaluation of one or multi-class classification error rate."""
            X_test = torch.as_tensor(X_test, device=device)
            y_test = check_numpy(y_test)
            self.model.train(False)
            with torch.no_grad():
                logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
                logits = check_numpy(logits)
                if logits.ndim == 1:
                    pred = (logits >= 0).astype(int)
                else:
                    pred = logits.argmax(axis=-1)
                error_rate = (y_test != pred).mean()
            return error_rate

    def evaluate_negative_auc(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)
            auc = roc_auc_score(y_test, logits)

        return -auc

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()
        error_rate = float(error_rate)  # To avoid annoying JSON unserializable bug
        return error_rate

    def evaluate_multiple_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean(axis=0)
        return error_rate.astype(float).tolist()

    def evaluate_ce_loss(self, X_test, y_test, device, batch_size=512):
        """Evaluate cross entropy loss for binary or multi-class targets.

        Args:
            X_test: input features.
            y_test (numpy Int array or torch Long tensor): the target classes.

        Returns:
            celoss (float): the average cross entropy loss.
        """
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = (process_in_chunks(self.model, X_test, batch_size=batch_size))
            y_test = torch.tensor(y_test, device=device)

            if logits.ndim == 1:
                celoss = F.binary_cross_entropy_with_logits(logits, y_test.float()).item()
            else:
                celoss = F.cross_entropy(logits, y_test).item()
        celoss = float(celoss)  # To avoid annoying JSON unserializable bug
        return celoss