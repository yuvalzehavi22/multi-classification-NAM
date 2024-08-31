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
                 monotonic_constraint: list = None):
        super().__init__(in_features, out_features)
        
        # Initialize LipschitzLinear layer, which already manages its weight and bias internally
        self.lipschitz_layer = lmn.LipschitzLinear(in_features, out_features, kind=weight_norms_kind)
        
        # Apply GroupSort as the activation function
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
    

#####################################################################################
################################### MODEL CLASSES ###################################
#####################################################################################

# FeatureNN Class
class FeatureNN_Base(torch.nn.Module):
    """
    Neural Network model for each individual feature.
    """
    def __init__(self,
                 num_units: int= 64,
                 hidden_units: list = [64, 32],
                 dropout: float = .5,
                 shallow: bool = True,
                 first_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,          
                 num_classes: int = 1,
                 ):
          
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in the first hidden layer.
          hidden_units: Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.
          dropout: Coefficient for dropout regularization.
          shallow: If True, then a shallow network with a single hidden layer is created (size: (1,num_units)),
                   otherwise, a network with more hidden layers is created (the number of hidden layers - hidden_units+1).
          first_layer: Activation and type of hidden unit (ExUs/ReLU) used in the first hidden layer.
          hidden_layer: Activation and type of hidden unit used in the next hidden layers (ReLULayer/MonotonicLayer),          
          num_classes: The output dimension of the feature block (adjusted to fit to muli-classification task)
        """
        super().__init__()
        
        self.num_units = num_units
        self.hidden_units = hidden_units
        self.dropout_val = dropout
        self.shallow = shallow
        self.activation_first_layer = first_layer
        self.activation_hidden_layer = hidden_layer
        self.num_classes = num_classes
        
        # First layer
        self.hidden_layers = torch.nn.ModuleList([
            self.activation_first_layer(1, self.num_units)
        ])
        
        in_units = self.num_units
        if not self.shallow:
            # Add hidden layers
            for out_units in hidden_units:
                self.hidden_layers.append(self.activation_hidden_layer(in_units, out_units))
                in_units = out_units  # Update in_units to the output of the last layer
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout) 
        
        # Last linear layer
        self.output_layer = torch.nn.Linear(in_units, self.num_classes, bias=False)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)      

    def forward(self, x):
        x = x.unsqueeze(1)

        # Pass through each hidden layer with dropout
        for layer in self.hidden_layers:
            x = layer(x)
            if self.dropout_val > 0.0:
                x = self.dropout(x)

        # Final output layer
        outputs = self.output_layer(x)
        
        return outputs
    

# FeatureNN Monotonic constrain Class
class FeatureNN_MonoBase(torch.nn.Module):
    """
    Neural Network model for each individual feature - constrain monotonic relationship.
    """
    def __init__(self,
                 num_units: int = 64,
                 hidden_units: list = [64, 32],
                 dropout: float = .5,
                 shallow: bool = True,
                 first_layer: ActivationLayer = LipschitzMonotonicLayer,
                 hidden_layer: ActivationLayer = LipschitzMonotonicLayer,          
                 num_classes: int = 1,
                 weight_norms_kind: str = "one-inf", 
                 group_size: int = 2, 
                 monotonic_constraint: list = None
                 ):
          
        """Initializes FeatureNN_MonoBase hyperparameters.

        Args:
          num_units: Number of hidden units in the first hidden layer.
          hidden_units: Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.
          dropout: Coefficient for dropout regularization.
          shallow: If True, then a shallow network with a single hidden layer is created (size: (1,num_units)),
                   otherwise, a network with more hidden layers is created (the number of hidden layers - hidden_units+1).
          first_layer: Activation and type of hidden unit (ExUs/ReLU) used in the first hidden layer.
          hidden_layer: Activation and type of hidden unit used in the next hidden layers (ReLULayer/MonotonicLayer),          
          num_classes: The output dimension of the feature block (adjusted to fit to muli-classification task)
          weight_norms_kind:  
          group_size: 
          monotonic_constraint: 
        """
        super().__init__()
        
        self.num_units = num_units
        self.hidden_units = hidden_units
        self.dropout_val = dropout
        self.shallow = shallow
        self.activation_first_layer = first_layer
        self.activation_hidden_layer = hidden_layer
        self.num_classes = num_classes
        
        #monotonic parameters
        self.weight_norms_kind_first_layer = weight_norms_kind
        self.activation_function_group_size = group_size
        self.monotonic_constraint = monotonic_constraint

        # First layer
        self.hidden_layers = torch.nn.ModuleList([
            self.activation_hidden_layer(1, 
                                         self.num_units, 
                                         weight_norms_kind=self.weight_norms_kind_first_layer, #"one-inf", 
                                         group_size=1, 
                                         monotonic_constraint=None)
        ])
        
        in_units = self.num_units
        if not self.shallow:
            # Add hidden layers
            for out_units in hidden_units:
                self.hidden_layers.append(
                    self.activation_hidden_layer(in_units, 
                                                 out_units,
                                                 weight_norms_kind="inf", 
                                                 group_size=1, #self.activation_function_group_size, 
                                                 monotonic_constraint=None)
                    )
                in_units = out_units  # Update in_units to the output of the last layer
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout) 
        
        # Last linear layer
        self.output_layer = self.activation_hidden_layer(in_units, 
                                         self.num_classes, 
                                         weight_norms_kind="inf", #"one-inf", 
                                         group_size=1, #self.activation_function_group_size, 
                                         monotonic_constraint=self.monotonic_constraint)      

    def forward(self, x):
        x = x.unsqueeze(1)

        # Pass through each hidden layer with dropout
        for layer in self.hidden_layers:
            x = layer(x)
            if self.dropout_val > 0.0:
                x = self.dropout(x)

        # Final output layer
        outputs = self.output_layer(x)
        
        return outputs
    

# FeatureNN block type Class
class FeatureNN(torch.nn.Module):
    """
    Neural network for individual features with configurable architecture.
    """
    def __init__(self,
                 num_units: int= 64,
                 hidden_units: list = [64, 32],
                 dropout: float = .5,
                 shallow: bool = True,
                 first_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,          
                 num_classes: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'

                 weight_norms_kind: str = "one-inf", 
                 group_size: int = 2, 
                 monotonic_constraint: list = None,
                 ):
        
        super().__init__()

        """Initializes FeatureNN hyperparameters.

        Args:
          Same as FeatureNN_Base - num_units, hidden_units, dropout, shallow, first_layer, hidden_layer, num_classes
          architecture_type: The architecture of FeatureNN ('single_to_multi_output', 'parallel_single_output', 'multi_output')
        """
        
        self.num_units = num_units
        self.hidden_units = hidden_units
        self.dropout_val = dropout
        self.shallow = shallow
        self.activation_first_layer = first_layer
        self.activation_hidden_layer = hidden_layer
        self.num_classes = num_classes
        self.architecture_type = architecture_type
      
        # Different architectures
        if self.architecture_type == 'multi_output':
            self.feature_nns = FeatureNN_Base(num_units= self.num_units,
                                        hidden_units = self.hidden_units,
                                        dropout = self.dropout_val,
                                        shallow = self.shallow,
                                        first_layer = self.activation_first_layer,
                                        hidden_layer = self.activation_hidden_layer,          
                                        num_classes = self.num_classes,
                                        )
            
        elif self.architecture_type == 'parallel_single_output': 
            self.feature_nns = torch.nn.ModuleList([
                    FeatureNN_Base(num_units= self.num_units,
                            hidden_units = self.hidden_units,
                            dropout = self.dropout_val,
                            shallow = self.shallow,
                            first_layer = self.activation_first_layer,
                            hidden_layer = self.activation_hidden_layer,          
                            num_classes = 1,
                            )
                    for i in range(self.num_classes)])
            
        elif self.architecture_type == 'single_to_multi_output':  
            self.feature_nns = FeatureNN_Base(num_units= self.num_units,
                                        hidden_units = self.hidden_units,
                                        dropout = self.dropout_val,
                                        shallow = self.shallow,
                                        first_layer = self.activation_first_layer,
                                        hidden_layer = self.activation_hidden_layer,          
                                        num_classes = 1,
                                        )
            
            self.output_layer = torch.nn.Linear(1, self.num_classes, bias=False)
            torch.nn.init.xavier_uniform_(self.output_layer.weight)

        elif self.architecture_type == 'monotonic_hidden_layer': #Enforce Monotonic Relationships
            #monotonic parameters
            self.weight_norms_kind_first_layer = weight_norms_kind
            self.activation_function_group_size = group_size
            self.monotonic_constraint = monotonic_constraint

            self.feature_nns = FeatureNN_MonoBase(num_units= self.num_units,
                                        hidden_units = self.hidden_units,
                                        dropout = self.dropout_val,
                                        shallow = self.shallow,
                                        first_layer = self.activation_first_layer,
                                        hidden_layer = self.activation_hidden_layer,          
                                        num_classes = 1,
                                        weight_norms_kind = self.weight_norms_kind_first_layer, 
                                        group_size = self.activation_function_group_size, 
                                        monotonic_constraint = self.monotonic_constraint,
                                        )

    def forward(self, x):
        
        if self.architecture_type == 'multi_output' or self.architecture_type == 'monotonic_hidden_layer':
            outputs = self.feature_nns(x)

        elif self.architecture_type == 'parallel_single_output':
            # Process each output separately using the corresponding branch
            single_output = [branch(x) for branch in self.feature_nns]
            outputs = torch.cat(single_output, dim=-1)
            
        elif self.architecture_type == 'single_to_multi_output':
            single_output = self.feature_nns(x)
            # Final output layer
            outputs = self.output_layer(single_output)
            
        return outputs


# Neural Additive Model (NAM) Class
class NeuralAdditiveModel(torch.nn.Module):
    """
    Combines multiple feature networks, each processing one feature, with dropout and bias
    """
    def __init__(self,
                 num_inputs: int,
                 num_units: int= 64,
                 hidden_units: list = [64, 32],
                 hidden_dropout: float = 0.,
                 feature_dropout: float = 0.,
                 shallow: bool = True,
                 first_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,          
                 num_classes: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 weight_norms_kind: str = "one-inf", 
                 group_size: int = 2, 
                 monotonic_constraint: list = None,
                 ):
        super().__init__()

        """Initializes NAM hyperparameters.

        Args:
          num_inputs: Number of feature inputs in input data.
          num_units: Number of hidden units in the first hidden layer.
          hidden_units: Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.
          hidden_dropout: Coefficient for dropout within each Feature NNs.
          feature_dropout: Coefficient for dropping out entire Feature NNs.
          shallow: If True, then a shallow network with a single hidden layer is created (size: (1,num_units)),
                   otherwise, a network with more hidden layers is created (the number of hidden layers - hidden_units+1).
          first_layer: Activation and type of hidden unit (ExUs/ReLU) used in the first hidden layer.
          hidden_layer: Activation and type of hidden unit used in the next hidden layers (ReLULayer/MonotonicLayer),          
          num_classes: The output dimension of the feature block (adjusted to fit to muli-classification task)
          architecture_type: The architecture of FeatureNN ('single_to_multi_output', 'parallel_single_output', 'multi_output')
          weight_norms_kind:  
          group_size: 
          monotonic_constraint: 
        """
        
        self.input_size = num_inputs

        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self.num_units = num_units
        elif isinstance(num_units, int):
            self.num_units = [num_units for _ in range(self.input_size)]

        self.hidden_units = hidden_units
        self.hidden_dropout = hidden_dropout
        self.feature_dropout_val = feature_dropout
        self.shallow = shallow
        self.activation_first_layer = first_layer
        self.activation_hidden_layer = hidden_layer
        self.num_classes = num_classes
        self.architecture_type = architecture_type

        if self.architecture_type == 'monotonic_hidden_layer': #Enforce Monotonic Relationships
            #monotonic parameters
            self.weight_norms_kind_first_layer = weight_norms_kind
            self.activation_function_group_size = group_size
            self.monotonic_constraint = monotonic_constraint
        
            self.feature_nns = torch.nn.ModuleList([
                                        FeatureNN(num_units=self.num_units[i],
                                                hidden_units = self.hidden_units,
                                                dropout = self.hidden_dropout,
                                                shallow = self.shallow,
                                                first_layer = self.activation_first_layer,
                                                hidden_layer = self.activation_hidden_layer,          
                                                num_classes = self.num_classes,
                                                architecture_type = self.architecture_type,
                                                weight_norms_kind = self.weight_norms_kind_first_layer, 
                                                group_size = self.activation_function_group_size, 
                                                monotonic_constraint = self.monotonic_constraint,
                                                ) 
                                        for i in range(self.input_size)])
        else:
            self.feature_nns = torch.nn.ModuleList([
                                        FeatureNN(num_units=self.num_units[i],
                                                hidden_units = self.hidden_units,
                                                dropout = self.hidden_dropout,
                                                shallow = self.shallow,
                                                first_layer = self.activation_first_layer,
                                                hidden_layer = self.activation_hidden_layer,          
                                                num_classes = self.num_classes,
                                                architecture_type = self.architecture_type) 
                                        for i in range(self.input_size)])
        
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(self.num_classes))
        
    def forward(self, x):
        # Collect outputs from each feature network
        FeatureNN_out = self._feature_nns(x)
        
        # Concatenates a sequence of tensors along the latent features dimension 
        f_out = torch.stack(FeatureNN_out, dim=-1)
        
        # Sum across features and add bias
        if self.feature_dropout_val > 0.0:
            f_out = self.feature_dropout(f_out)

        outputs = f_out.sum(axis=-1) + self.bias #if I want positive bias: F.relu(self.bias)
        
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
                 num_inputs: int,
                 #phase1 - latent_features:
                 num_units_phase1: int= 64,
                 hidden_units_phase1: list = [64, 32],
                 hidden_dropout_phase1: float = 0.,
                 feature_dropout_phase1: float = 0.,
                 shallow_phase1: bool = True,
                 first_layer_phase1: ActivationLayer = ExULayer,
                 hidden_layer_phase1: ActivationLayer = ReLULayer,          
                 latent_var_dim: int = 1,
                 featureNN_architecture_phase1: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 weight_norms_kind_phase1: str = "one-inf", 
                 group_size_phase1: int = 2, 
                 monotonic_constraint_phase1: list = None,
                 #phase2 - final outputs:
                 num_units_phase2: int= 64,
                 hidden_units_phase2: list = [64, 32],
                 hidden_dropout_phase2: float = 0.,
                 feature_dropout_phase2: float = 0.,
                 shallow_phase2: bool = True,
                 first_layer_phase2: ActivationLayer = ExULayer,
                 hidden_layer_phase2: ActivationLayer = ReLULayer,          
                 output_dim: int = 1,
                 featureNN_architecture_phase2: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 weight_norms_kind_phase2: str = "one-inf", 
                 group_size_phase2: int = 2, 
                 monotonic_constraint_phase2: list = None,
                 ):
        super().__init__()

        """Initializes NAM hyperparameters.

        Args:
          1) num_inputs: Number of feature inputs in input data.

          Parameter types are common to both layers of the NAM (phase1 & phase2)
          2) num_units: Number of hidden units in the first hidden layer.
          3) hidden_units: Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.
          4) hidden_dropout: Coefficient for dropout within each Feature NNs.
          5) feature_dropout: Coefficient for dropping out entire Feature NNs.
          6) shallow: If True, then a shallow network with a single hidden layer is created (size: (1,num_units)),
                   otherwise, a network with more hidden layers is created (the number of hidden layers - hidden_units+1).
          7) first_layer: Activation and type of hidden unit (ExUs/ReLU) used in the first hidden layer.
          8) hidden_layer: Activation and type of hidden unit used in the next hidden layers (ReLULayer/MonotonicLayer),          
          9) architecture_type: The architecture of FeatureNN ('single_to_multi_output', 'parallel_single_output', 'multi_output')

          10) latent_var_dim: The output dimension of the feature block for the first phase
          11) output_dim: The output dimension of the feature block for the second phase - this is the number of classes in the classification task

        """

        self.input_size = num_inputs

        # phase1 hyperparameters - latent_features:
        self.num_units_phase1 = num_units_phase1
        self.hidden_units_phase1 = hidden_units_phase1
        self.hidden_dropout_phase1 = hidden_dropout_phase1
        self.feature_dropout_phase1 = feature_dropout_phase1
        self.shallow_phase1 = shallow_phase1
        self.activation_first_layer_phase1 = first_layer_phase1
        self.activation_hidden_layer_phase1 = hidden_layer_phase1
        self.latent_var_dim = latent_var_dim
        self.architecture_type_phase1 = featureNN_architecture_phase1
        self.weight_norms_kind_first_layer_phase1 = weight_norms_kind_phase1
        self.activation_function_group_size_phase1 = group_size_phase1
        self.monotonic_constraint_phase1 = monotonic_constraint_phase1

        # phase1 hyperparameters - latent_features:
        self.num_units_phase2 = num_units_phase2
        self.hidden_units_phase2 = hidden_units_phase2
        self.hidden_dropout_phase2 = hidden_dropout_phase2
        self.feature_dropout_phase2 = feature_dropout_phase2
        self.shallow_phase2 = shallow_phase2
        self.activation_first_layer_phase2 = first_layer_phase2
        self.activation_hidden_layer_phase2 = hidden_layer_phase2
        self.num_classes = output_dim
        self.architecture_type_phase2 = featureNN_architecture_phase2
        self.weight_norms_kind_first_layer_phase2 = weight_norms_kind_phase2
        self.activation_function_group_size_phase2 = group_size_phase2
        self.monotonic_constraint_phase2 = monotonic_constraint_phase2

        self.NAM_features = NeuralAdditiveModel(
                                num_inputs = self.input_size,
                                num_units = self.num_units_phase1,
                                hidden_units = self.hidden_units_phase1,
                                hidden_dropout = self.hidden_dropout_phase1,
                                feature_dropout = self.feature_dropout_phase1,
                                shallow = self.shallow_phase1,
                                first_layer = self.activation_first_layer_phase1,
                                hidden_layer = self.activation_hidden_layer_phase1,
                                num_classes = self.latent_var_dim,
                                architecture_type = self.architecture_type_phase1,
                                weight_norms_kind = self.weight_norms_kind_first_layer_phase1, 
                                group_size = self.activation_function_group_size_phase1, 
                                monotonic_constraint = self.monotonic_constraint_phase1,          
                                )

        self.NAM_output = NeuralAdditiveModel(
                                num_inputs = self.latent_var_dim,
                                num_units = self.num_units_phase2,
                                hidden_units = self.hidden_units_phase2,
                                hidden_dropout = self.hidden_dropout_phase2,
                                feature_dropout = self.feature_dropout_phase2,
                                shallow = self.shallow_phase2,
                                first_layer = self.activation_first_layer_phase2,
                                hidden_layer = self.activation_hidden_layer_phase2,
                                num_classes = self.num_classes,
                                architecture_type = self.architecture_type_phase2,
                                weight_norms_kind = self.weight_norms_kind_first_layer_phase2, 
                                group_size = self.activation_function_group_size_phase2, 
                                monotonic_constraint = self.monotonic_constraint_phase2,   
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