import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions.uniform import Uniform
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    @staticmethod
    def get_synthetic_data_phase1(num_exp=10, in_features=10, is_test=False):
        """
        Generate synthetic data for Phase 1.
        
        Parameters:
        -----------
        num_exp : int
            Number of experiments (samples).
        
        in_features : int
            Number of input features.

        is_test : bool
            generate data for testing the results

        Returns:
        --------
        X : torch.Tensor
            Generated input features.
        
        y : torch.Tensor
            Generated target values.
        """
        # Simulate independent variables, x0,...,xn from a Uniform distribution on [0, 3]
        if is_test:
            x_values = torch.linspace(0, 3, num_exp).reshape(-1, 1)  # 100 points between -1 and 1
            X = x_values.repeat(1, in_features)
        else:
            X = Uniform(0, 3).sample((num_exp, in_features))
        print(X.shape)
        
        # Creating a dict to save all the true shape functions 
        shape_functions = {}
        for j in range(4):
            for i in range(in_features):
                shape_functions[f"f_{j}_{i}"] = torch.zeros(num_exp)

        # creating y_0
        shape_functions['f_0_0'] = (3/2)*X[:, 0]
        shape_functions['f_0_1'] = (1/4)*(X[:, 2]**3)
        shape_functions['f_0_2'] = (1/3)*(X[:, 1]**2)
        y_0 = shape_functions['f_0_0'] + shape_functions['f_0_1'] + shape_functions['f_0_2']
        y_0 = y_0.reshape(-1, 1)
        
        # creating y_1
        shape_functions['f_1_5'] = 5*torch.exp(-4 * X[:, 7].abs())
        shape_functions['f_1_6'] = 2*torch.cos(5 * X[:, 6])
        y_1 = shape_functions['f_1_5'] + shape_functions['f_1_6']
        y_1 = y_1.reshape(-1, 1)
        
        # creating y_2
        shape_functions['f_2_7'] = (2/3)*torch.log(100 * X[:, 5].abs()+1)
        shape_functions['f_2_8'] = (5/2)*(torch.sin(5 * X[:, 8])+1)
        y_2 = shape_functions['f_2_7'] + shape_functions['f_2_8']
        y_2 = y_2.reshape(-1, 1)
        
        # creating y_3
        shape_functions['f_3_5'] = 3*torch.exp(-4 * X[:, 7].abs())
        shape_functions['f_3_2'] = 0.5*(X[:, 2]**2)
        y_3 = shape_functions['f_3_5'] + shape_functions['f_3_2']
        y_3 = y_3.reshape(-1, 1)
        
        # Stack all y_i to form the final target matrix
        y = torch.cat([y_0 ,y_1, y_2, y_3], dim=1)
        print(y.shape)
        
        return X, y, shape_functions

    @staticmethod
    def get_synthetic_data_phase2(X_input, is_test=False):
        """
        Generate synthetic target values for Phase 2 using input features.
        
        Parameters:
        -----------
        X_input : torch.Tensor
            Input features for generating synthetic targets.
        
        is_test : bool
            generate data for testing the results.
        
        Returns:
        --------
        y : torch.Tensor
            Generated target values for Phase 2.
        """
        if is_test:
            x_values = torch.linspace(0, 3, X_input.size(0)).reshape(-1, 1)  # 100 points between -1 and 1
            X_input = x_values.repeat(1, X_input.size(1))

        # Creating a dict to save all the true shape functions 
        shape_functions = {}
        for j in range(2):
            for i in range(X_input.size(1)):
                shape_functions[f"f_{j}_{i}"] = torch.zeros(X_input.size(0))

        # creating y_0
        shape_functions['f_0_0'] = X_input[:, 0]
        shape_functions['f_0_2'] = 2*X_input[:, 2]
        y_0 = shape_functions['f_0_0'] + shape_functions['f_0_2']
        y_0 = y_0.reshape(-1, 1)
        
        # creating y_1
        shape_functions['f_1_1'] = 3*X_input[:, 1]
        y_1 = shape_functions['f_1_1']
        y_1 = y_1.reshape(-1, 1)
        
        # Stack all y_i to form the final target matrix
        y = torch.cat([y_0, y_1], dim=1)
        print(y.shape)

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