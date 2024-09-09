import torch
import torch.nn as nn

class CustomModel(nn.Module):
    """
    A flexible model class that can adapt to regression, binary classification, and multi-class classification tasks.
    
    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output units.
    task_type : str
        The type of task. Options: 'regression', 'binary_classification', 'multi_classification'.
    """

    def __init__(self, task_type='regression'):
        super(CustomModel, self).__init__()
        
        self.task_type = task_type
        
        # Define activation based on task type
        if self.task_type == 'binary_classification':
            self.final_activation = nn.Sigmoid()
        elif self.task_type == 'multi_classification':
            self.final_activation = nn.Softmax(dim=1)  # Softmax for multi-class classification
        else:
            self.final_activation = None  # No activation for regression (linear output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply activation based on the task
        if self.activation:
            x = self.activation(x)
        
        return x

    def get_loss_fn(self):
        """
        Returns the appropriate loss function based on the task type.
        """
        if self.task_type == 'regression':
            return nn.MSELoss()
        elif self.task_type == 'binary_classification':
            return nn.BCELoss()  # Binary Cross-Entropy Loss
        elif self.task_type == 'multi_classification':
            return nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

