import numpy as np
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import math

from mononet import MonotonicLayer

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
        self.bias = torch.nn.Parameter(torch.empty(in_features)) #before it was in_features - understand why!

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
    

#####################################################################################
################################### MODEL CLASSES ###################################
#####################################################################################

# FeatureNN Class
class FeatureNN(torch.nn.Module):
    """
    Neural network for individual features with configurable architecture.
    """
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 output_dim: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 ):
        
        super().__init__()
        
        self.architecture_type = architecture_type
        
        # First (shallow) layer
        self.shallow_layer = shallow_layer(1, shallow_units)
        
        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        in_units = shallow_units
        for out_units in hidden_units:
            self.hidden_layers.append(hidden_layer(in_units, out_units))
            in_units = out_units  # Update in_units to the output of the last layer
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout) 
        
        # Different architectures
        if self.architecture_type == 'multi_output':
            self.output_layer = torch.nn.Linear(in_units, output_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.output_layer.weight)
            
        elif self.architecture_type == 'parallel_single_output' or self.architecture_type == 'single_to_multi_output':
            self.output_layer = torch.nn.Linear(in_units, 1, bias=False)
            torch.nn.init.xavier_uniform_(self.output_layer.weight)
            

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Pass through the shallow layer
        x = self.shallow_layer(x)
        #x = self.dropout(x)

        # Pass through each hidden layer with dropout
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout(x)

        # Final output layer
        outputs = self.output_layer(x)
        
        return outputs
    

# FeatureNN Class - 



# FeatureNN block type Class
class FeatureNN_BlockType(torch.nn.Module):
    """
    Neural network for individual features with configurable architecture.
    """
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 output_dim: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output'
                 ):
        
        super().__init__()
        
        self.architecture_type = architecture_type
        self.num_classes = output_dim
       
        # Different architectures
        if self.architecture_type == 'multi_output':
            self.feature_nns = FeatureNN(shallow_units=shallow_units,
                                         hidden_units=hidden_units,
                                         shallow_layer=shallow_layer,
                                         hidden_layer=hidden_layer,
                                         dropout=dropout,
                                         output_dim=output_dim,
                                         architecture_type=architecture_type)
            
        elif self.architecture_type == 'parallel_single_output': 
            self.feature_nns = torch.nn.ModuleList([
                    FeatureNN(shallow_units=shallow_units,
                              hidden_units=hidden_units,
                              shallow_layer=shallow_layer,
                              hidden_layer=hidden_layer,
                              dropout=dropout,
                              output_dim=output_dim, 
                              architecture_type=architecture_type)
                    for i in range(output_dim)])
            
        elif self.architecture_type == 'single_to_multi_output':  
            self.feature_nns = FeatureNN(shallow_units=shallow_units,
                                         hidden_units=hidden_units,
                                         shallow_layer=shallow_layer,
                                         hidden_layer=hidden_layer,
                                         dropout=dropout,
                                         output_dim=output_dim,
                                         architecture_type=architecture_type
                                        )
            
            self.output_layer = torch.nn.Linear(1, output_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.output_layer.weight)

        # elif self.architecture_type == 'monotonic_hidden_layer': #Enforce Monotonic Relationships
        #     self.feature_nns = FeatureNN(shallow_units=shallow_units,
        #                                  hidden_units=hidden_units,
        #                                  shallow_layer=shallow_layer,
        #                                  hidden_layer=hidden_layer,
        #                                  dropout=dropout,
        #                                  output_dim=output_dim,
        #                                  architecture_type=architecture_type
        #                                 )
            
            # # Apply monotonic constraints directly after the FeatureNN
            # self.monotonic = torch.nn.Sequential(
            #     MonotonicLayer(hidden_units[-1], 32, fn='tanh_p1'),
            #     torch.nn.LeakyReLU(),
            #     MonotonicLayer(32, 16, fn='tanh_p1'),
            #     torch.nn.LeakyReLU(),
            #     MonotonicLayer(16, output_dim, fn='tanh_p1'),
            # )

    def forward(self, x):
        
        if self.architecture_type == 'multi_output':
            outputs = self.feature_nns(x)

        elif self.architecture_type == 'parallel_single_output':
            # Process each output separately using the corresponding branch
            single_output = [branch(x) for branch in self.feature_nns]
            outputs = torch.cat(single_output, dim=-1)
            
        elif self.architecture_type == 'single_to_multi_output':
            single_output = self.feature_nns(x)
            # Final output layer
            outputs = self.output_layer(single_output)

        elif self.architecture_type == 'monotonic_hidden_layer':
            hidden_output = self.feature_nns(x)
            outputs = self.monotonic(hidden_output)
            
        return outputs


# Neural Additive Model (NAM) Class
class NeuralAdditiveModel(torch.nn.Module):
    """
    Combines multiple feature networks, each processing one feature, with dropout and bias
    """
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 hidden_dropout: float = 0.,
                 feature_dropout: float = 0.,
                 output_dim: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output'
                 ):
        super().__init__()
        
        self.input_size = input_size

        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN_BlockType(shallow_units=shallow_units[i],
                                hidden_units=hidden_units,
                                shallow_layer=shallow_layer,
                                hidden_layer=hidden_layer,
                                dropout=hidden_dropout,
                                output_dim=output_dim,
                                architecture_type=architecture_type) 
            for i in range(input_size)])
        
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        # Collect outputs from each feature network
        FeatureNN_out = self._feature_nns(x)
        
        # Concatenates a sequence of tensors along the latent features dimension 
        f_out = torch.stack(FeatureNN_out, dim=-1)
        
        # Sum across features and add bias
        f_out = self.feature_dropout(f_out)
        outputs = f_out.sum(axis=-1) + self.bias
        
        if 0:
            print('final output', outputs)
            print('f_out', f_out)
        return outputs, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]
    

# Hirarchical Neural Additive Model Class
class HierarchNeuralAdditiveModel(torch.nn.Module):
    """
    Hierarch Neural Additive Model
    """
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 hidden_dropout: float = 0.,
                 feature_dropout: float = 0.,
                 latent_feature_dropout: float = 0.,
                 latent_var_dim: int = 1,
                 output_dim: int = 1,
                 featureNN_architecture_phase1: str = 'multi_output',
                 featureNN_architecture_phase2: str = 'multi_output',
                 ):
        super().__init__()

        self.NAM_features = NeuralAdditiveModel(input_size=input_size,
                                shallow_units= shallow_units,
                                hidden_units= hidden_units,
                                shallow_layer= shallow_layer,
                                hidden_layer= hidden_layer,
                                hidden_dropout= hidden_dropout,
                                feature_dropout= feature_dropout,
                                output_dim= latent_var_dim,
                                architecture_type= featureNN_architecture_phase1,                
                                )
       

        self.NAM_output = NeuralAdditiveModel(input_size=latent_var_dim,
                                shallow_units= shallow_units,
                                hidden_units= hidden_units,
                                shallow_layer= shallow_layer,
                                hidden_layer= hidden_layer,
                                hidden_dropout= hidden_dropout,
                                feature_dropout= latent_feature_dropout,
                                output_dim = output_dim,
                                architecture_type= featureNN_architecture_phase2,
                                )

    def forward(self, x):
        
        latent_outputs, f_out = self.NAM_features(x)

        outputs, lat_f_out = self.NAM_output(latent_outputs)
       
         # Apply softmax to get class probabilities
#         outputs = torch.softmax(outputs, dim=-1)

        if 0:
            print('x:', x.shape)
            print('latent_outputs:',latent_outputs.shape)
            print('f_out:',f_out.shape)
            print('outputs:',outputs.shape)
            print('lat_f_out:',lat_f_out.shape)  
            
        return outputs, lat_f_out