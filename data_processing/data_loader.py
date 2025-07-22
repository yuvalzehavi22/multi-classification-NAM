import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions.uniform import Uniform
from torch.distributions import Normal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import os

import wandb

class TorchDataset(Dataset):
    """
    Dataset class for PyTorch. This supports numpy arrays for both inputs and targets.

    Parameters
    ----------
    x : 2D array
        The input matrix
    y : 1D array
        The target values
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DataLoaderWrapper:
    """
    Wrapper class for handling data loading, including validation sets.

    Parameters
    ----------
    X : np.ndarray
        The input features
    y : np.ndarray
        The target labels
    batch_size : int
        Number of samples per batch
    num_workers : int
        Number of subprocesses to use for data loading
    val_split : float
        Proportion of data to be used for validation set
    shuffle : bool
        Whether to shuffle the data
    """

    def __init__(self, X, y, batch_size=32, num_workers=0, val_split=0.2, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle

        # Split into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=self.val_split, shuffle=self.shuffle, random_state=42
        )

    def get_data_size(self):
        """
        Returns the size of the training and validation data.
        """
        train_size = len(self.X_train)
        val_size = len(self.X_val)
        return train_size, val_size

    def create_dataloaders(self):
        """
        Creates and returns PyTorch DataLoaders for training and validation sets.

        Returns
        -------
        train_dataloader : torch.utils.data.DataLoader
        val_dataloader : torch.utils.data.DataLoader
        """
        # Create training dataset and dataloader
        train_dataset = TorchDataset(self.X_train, self.y_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

        # Create validation dataset and dataloader
        val_dataset = TorchDataset(self.X_val, self.y_val)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers
        )

        return train_dataloader, val_dataloader


# Generating synthetic dataset
class SyntheticDatasetGenerator:
    """
    Class for generating synthetic datasets for different phases and creating DataLoaders.

    Methods:
    --------
    get_synthetic_data_phase1(num_exp=10, in_features=10):
        Generates synthetic dataset for Phase 1 based on a combination of uniform and normal distributions.
    
    get_synthetic_data_phase2(X_input):
        Generates synthetic targets for Phase 2 based on existing input data.
    
    make_loader(X, y, batch_size):
        Creates a DataLoader from given input features and target values.
    """
    # @staticmethod
    # def get_synthetic_data_phase1(num_exp=2000, raw_features=5, num_concepts=3, is_test=False, seed=42, X_val=None):
    #     torch.manual_seed(seed)
    #     if is_test:
    #         x_values = torch.linspace(0, 3, num_exp).reshape(-1, 1)
    #         X = x_values.repeat(1, raw_features)
    #     elif X_val is not None:
    #         X = X_val
    #     else:
    #         X = Uniform(0, 3).sample((num_exp, raw_features))

    #     shape_functions = {f'f_{j}_{i}': torch.zeros_like(X[:, 0]) for j in range(num_concepts) for i in range(raw_features)}
    #     out_weights = {}

    #     # Isolated, interpretable concept functions using separate features
    #     concepts = []
    #     for j in range(num_concepts):
    #         feature_idx = j  # Use feature 0, 1, 2 → concept 0, 1, 2
    #         key = f"f_{j}_{feature_idx}"
    #         x = X[:, feature_idx]

    #         if j == 0:
    #             shape = torch.sin(2 * x)
    #         elif j == 1:
    #             shape = x ** 2
    #         elif j == 2:
    #             shape = torch.exp(0.3 * x)
    #         else:
    #             shape = torch.zeros_like(x)

    #         shape_functions[key] = shape
    #         out_weights[key] = 1.0  # full attribution
    #         concepts.append(shape.unsqueeze(1))

    #     concepts = torch.cat(concepts, dim=1)
    #     return X, concepts, shape_functions, out_weights

    # @staticmethod
    # def get_synthetic_data_phase2(concepts, num_classes=4, is_test=False, device=None):
    #     if device is None:
    #         device = concepts.device

    #     if is_test:
    #         x_values = torch.linspace(round(float(concepts.min())), round(float(concepts.max())), concepts.size(0), device=device).reshape(-1, 1)
    #         concepts = x_values.repeat(1, concepts.size(1)).to(device)

    #     shape_functions = {f'f_{j}_{i}': torch.zeros_like(concepts[:, 0]) for j in range(num_classes) for i in range(concepts.size(1))}
    #     outputs = []

    #     for j in range(num_classes):
    #         c0, c1, c2 = concepts[:, 0], concepts[:, 1], concepts[:, 2]

    #         if j == 0:
    #             out = 0.7 * c0
    #         elif j == 1:
    #             out = 0.5 * c1
    #         elif j == 2:
    #             out = 0.6 * c2
    #         elif j == 3:
    #             out = 0.4 * torch.tanh(3 * c1) * c1 ** 2
    #         else:
    #             out = torch.zeros_like(c0)

    #         outputs.append(out.unsqueeze(1))
    #         for i in range(concepts.size(1)):
    #             key = f"f_{j}_{i}"
    #             if j == 0 and i == 0:
    #                 shape_functions[key] = 0.7 * c0
    #             elif j == 1 and i == 1:
    #                 shape_functions[key] = 0.5 * c1
    #             elif j == 2 and i == 2:
    #                 shape_functions[key] = 0.6 * c2
    #             elif j == 3 and i == 1:
    #                 shape_functions[key] = 0.4 * torch.tanh(3 * c1) * c1 ** 2
    #             else:
    #                 shape_functions[key] = torch.zeros_like(c0)

    #     y = torch.cat(outputs, dim=1)
    #     return y, shape_functions
    
    @staticmethod
    def get_synthetic_data_phase1(num_exp=10, raw_features=10, num_concepts=4, is_test=False, seed=0, X_val=None):
        # Set a seed for reproducibility
        torch.manual_seed(seed)

        # Simulate independent variables, x0,...,xn from a Uniform distribution on [0, 3]
        if is_test:
            x_values = torch.linspace(0, 3, num_exp).reshape(-1, 1)
            X = x_values.repeat(1, raw_features)
        elif X_val is not None: # only for learn_only_concept_to_target param - I already created X
            X = X_val
        else:
            X = Uniform(0, 3).sample((num_exp, raw_features))

        # Initialize dictionaries for shape functions and weights
        shape_functions = {}
        out_weights = {}
        for j in range(num_concepts):
            for i in range(raw_features):
                shape_functions[f"f_{j}_{i}"] = torch.zeros(num_exp)
                out_weights[f"f_{j}_{i}"] = 0

        # Generate weights and shape functions for each concept
        for j in range(num_concepts):
            # Select consistent feature indices for each concept
            selected_features = torch.arange(raw_features)[j::num_concepts]
            
            # Add additional unique indices if needed
            if len(selected_features) < 3:
                existing_indices = set(selected_features.tolist())
                last_item = (selected_features[-1] + num_concepts) % raw_features
                additional_indices = []
                
                for offset in range(3 - len(selected_features)):
                    new_item = (last_item + offset * num_concepts) % raw_features
                    
                    # Ensure `new_item` is unique by finding the next available index
                    while int(new_item) in existing_indices:
                        new_item = (new_item + 1) % raw_features  # Increment and wrap around if necessary
                    
                    if int(new_item) not in existing_indices:
                        additional_indices.append(int(new_item))
                        existing_indices.add(int(new_item))

                selected_features = torch.cat((selected_features, torch.tensor(additional_indices)))

            # print('selected_features:')
            # print(selected_features)

            for i, feature_idx in enumerate(selected_features):
                key = f"f_{j}_{feature_idx}"

                # Assign weights with more conditional cases for variety
                if i % 2 == 0 and j % 2 == 0:
                    out_weights[key] = (i + 2) * 0.5 / (j + 1)
                elif i % 2 == 1 and j % 2 == 1:
                    out_weights[key] = (int(feature_idx) + 1) * 0.1 * (j + 1)
                elif int(feature_idx) % 2 == 0:
                    out_weights[key] = (i + 1) * 0.25 / (j + 2)
                else:
                    out_weights[key] = (int(feature_idx) + 3) * 0.15

                # Create shape functions dynamically
                if int(feature_idx) == 0:
                    shape_functions[key] = out_weights[key] * X[:, feature_idx]  # Linear function
                elif int(feature_idx) == 1:
                    shape_functions[key] = out_weights[key] * (torch.cos(4 * X[:, feature_idx])+1)
                elif int(feature_idx) == 2:
                    shape_functions[key] = out_weights[key] * (X[:, feature_idx]**2)  # Quadratic
                elif int(feature_idx) == 3:
                    shape_functions[key] = out_weights[key] * torch.exp(0.3*X[:, feature_idx])  # Exponential
                elif int(feature_idx) == 4:
                    out_weights[key] = 0
                    shape_functions[key] = out_weights[key] * X[:, feature_idx]
                elif int(feature_idx) == 5:
                    shape_functions[key] = out_weights[key] * (X[:, feature_idx]**3)
                elif int(feature_idx) == 6:
                    shape_functions[key] = out_weights[key] * (torch.sin(5 * X[:, feature_idx])+1)
                elif int(feature_idx) == 7 or int(feature_idx) == 8:
                    out_weights[key] = 0
                    shape_functions[key] = out_weights[key] * X[:, feature_idx]
                else:
                    shape_functions[key] = out_weights[key] * (torch.log(X[:, feature_idx]+1))

            # Sum up shape functions to form a concept
            concept = sum(shape_functions[f"f_{j}_{feature_idx}"] for feature_idx in selected_features)
            concept = concept.reshape(-1, 1)

            # Concatenate the concept to the concepts matrix
            if j == 0:
                concepts = concept
            else:
                concepts = torch.cat((concepts, concept), dim=1)

        return X, concepts, shape_functions, out_weights

    @staticmethod      
    def get_synthetic_data_phase2(concepts, num_classes=2, is_test=False, device=None):
        """
        Generate synthetic target values for Phase 2 using input features.
        
        Parameters:
        -----------
        concepts : torch.Tensor
            Input for the layer - the concepts for generating synthetic targets.
        
        is_test : bool
            generate data for testing the results.
        
        Returns:
        --------
        y : torch.Tensor
            Generated target values for Phase 2.
        """
        if device is None:
            device = concepts.device

        if is_test:
            x_values = torch.linspace(round(float(concepts.min())), round(float(concepts.max())), concepts.size(0), device=device).reshape(-1, 1) 
            concepts = x_values.repeat(1, concepts.size(1)).to(device)

        # Creating a dict to save all the true shape functions 
        shape_functions = {}

        # Generate y_i for each output class
        outputs = []
        #math_expressions = []

        for j in range(num_classes):
            output_sum = torch.zeros(concepts.size(0), device=device)

            # Assign unique weights for this class
            class_weights = [0.5 + 0.1 * j, 0.3 + 0.05 * j, 0.2 + 0.02 * j, 0.1 + 0.03 * j]
            #expression = f"output_{j} = "

            for i in range(concepts.size(1)):
                key = f"f_{j}_{i}"

                # Choose different function types for each concept-to-output mapping
                if (j + i) % 3 == 0:
                    shape_functions[key] = class_weights[0] * concepts[:, i]  # Linear
                    #expression += f"{class_weights[0]:.2f} * concept[:, {i}] + "
                elif (j + i) % 3 == 1:
                    shape_functions[key] = class_weights[1] * torch.exp(0.2 * concepts[:, i])  # Exponential
                    #expression += f"{class_weights[1]:.2f} * exp(0.2 * concept[:, {i}]) + "
                elif (j + i) % 3 == 2:
                    shape_functions[key] = class_weights[2] * torch.tanh(5.0 * concepts[:, i])*(concepts[:, i] ** 2)  # Quadratic
                    #expression += f"{class_weights[2]:.2f} * (concept[:, {i}] ** 2) + "
                else:
                    shape_functions[key] = class_weights[3] * (concepts[:, i]**3) #torch.sin(0.5 * concepts[:, i])  # Sinusoidal
                    #expression += f"{class_weights[3]:.2f} * (concept[:, {i}] ** 3) + "
                
                output_sum += shape_functions[key]  # Add the function to the current output sum

            # expression = expression.rstrip(' + ')  # Remove trailing ' + '
            # math_expressions.append(expression)
            outputs.append(output_sum.reshape(-1, 1))

        # Stack all outputs to form the final target matrix
        y = torch.cat(outputs, dim=1)

        # # Print the generated math expressions for each output
        # for exp in math_expressions:
        #     print(exp)

        return y, shape_functions

    @staticmethod
    def make_loader(X, y, batch_size=32):
        """
        Create a DataLoader from input features and target values.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input features.
        
        y : torch.Tensor
            Target values.
        
        batch_size : int, optional (default=32)
            Number of samples per batch.

        Returns:
        --------
        DataLoader
            PyTorch DataLoader for the dataset.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    
