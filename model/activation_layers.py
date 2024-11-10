import numpy as np
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import math
import monotonicnetworks as lmn

def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    """
    Initializes a tensor with values from a truncated normal distribution
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

# ActivationLayer Class
class ActivationLayer(torch.nn.Module):
    """
    Abstract base class for layers with weights and biases
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    """
    Custom layer using exponential activation with weight and bias initialization
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x): 
        # First, multiply the input by the exponential of the weights
        exu = (x - self.bias) @ torch.exp(self.weight)
        output = torch.clip(exu, 0, 1)
        
        if 0:
            print('ExULayer_weights:', self.weight.detach().cpu().numpy())
            print('ExULayer Normalization L1\n:', torch.linalg.norm(self.weight.t(), 1, dim=0))
            print('ExULayer Normalization L2\n:',torch.linalg.norm(self.weight.t(), 2, dim=0))
        
        return output


class ReLULayer(ActivationLayer):
    """
    Custom layer using ReLU activation with Xavier weight initialization
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        output = F.relu((x - self.bias) @ self.weight)
        
        if 0:
            print('ReLULayer_weights:', self.weight.detach().cpu().numpy())
            print('ReLULayer Normalization L1:\n', torch.linalg.norm(self.weight.t(), 1, dim=0))
            print('ReLULayer Normalization L2:\n',torch.linalg.norm(self.weight.t(), 2, dim=0))
        
        return output
    

class MonotonicLayer(ActivationLayer):

    def __init__(self, 
                 in_features: int, 
                 out_features: int):
        super().__init__(in_features, out_features)

        torch.nn.init.normal_(self.weight, mean=0)

        self.fn = 'tanh_p1'
        self.pos_fn = self.pos_tanh
        
    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def forward(self, input: torch):
        ret = torch.matmul(input, self.pos_fn(self.weight))

        if 0:
            print(f"Input shape: {input.shape}")
            print(f"Weight shape: {self.pos_fn(self.weight).shape}")
        return ret


class LipschitzMonotonicLayer(ActivationLayer):
    """
    A layer that combines Lipschitz constraints with monotonicity, inheriting from ActivationLayer
    to ensure weights and biases are handled as model parameters.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 weight_norms_kind: str = "one-inf", 
                 group_size: int = 2, 
                 monotonic_constraint = None):
        
        super().__init__(in_features, out_features)
        
        # Initialize LipschitzLinear layer, which already manages its weight and bias internally
        self.lipschitz_layer = lmn.LipschitzLinear(in_features, out_features, kind=weight_norms_kind)
        
        # Apply GroupSort as the activation function
        # Ensure group_size is compatible with the number of features
        if group_size > out_features:
            group_size = out_features  # Fallback to using the total number of features if group_size is too large
        
        self.activation = lmn.GroupSort(group_size)
        
        # Combine the layers and apply monotonic constraint if specified
        if monotonic_constraint is not None:
            self.layer = lmn.MonotonicWrapper(
                torch.nn.Sequential(self.lipschitz_layer, self.activation),
                monotonic_constraints=monotonic_constraint
            )
        else:
            self.layer = torch.nn.Sequential(self.lipschitz_layer, self.activation)

    def forward(self, x):
        # The forward pass applies the layers in sequence
        return self.layer(x)