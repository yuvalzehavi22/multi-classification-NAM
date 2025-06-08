import numpy as np
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import math
from data_processing.data_loader import SyntheticDatasetGenerator
from model.activation_layers import *

# FeatureNN Class
class FeatureNN_Base(torch.nn.Module):
    """
    Neural Network model for each individual feature, adaptable for different activation functions and architectures.
    """
    def __init__(self,
                 num_units: int= 64,
                 hidden_units: list = [64, 32],
                 dropout: float = 0.5,
                 shallow: bool = True,
                 first_layer: str = 'ReLU',
                 hidden_layer: str = 'ReLU',
                 num_classes: int = 1,
                 ):
          
        """
        Initializes FeatureNN perparameters.

        Perparameters:
        --------------
          num_units: Number of hidden units in the first hidden layer.
          hidden_units: Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.
          dropout: Coefficient for dropout regularization.
          shallow: If True, then a shallow network with a single hidden layer is created (size: (1,num_units)),
                   otherwise, a network with more hidden layers is created (the number of hidden layers - hidden_units+1).
          first_layer: Activation and type of hidden unit (ExUs/ReLU) used in the first hidden layer.
          hidden_layer: Activation and type of hidden unit used in the next hidden layers (ReLULayer/MonotonicLayer),          
          num_classes: The output dimension of the feature block (adjusted to fit to muli-classification task)
        """

        super(FeatureNN_Base, self).__init__()
        
        self.num_units = num_units
        self.hidden_units = hidden_units
        self.dropout_val = dropout
        self.num_classes = num_classes
        self.shallow = shallow

        # Initialize dropout layer
        self.dropout = torch.nn.Dropout(p=self.dropout_val)

        # Set activation functions for first and hidden layers
        self.activation_first_layer = self._get_activation(first_layer, 'first')
        self.activation_hidden_layer = self._get_activation(hidden_layer, 'hidden')
        
        # Create the network layers
        self._initialize_layers()


    def _get_activation(self, layer_name: str, layer_type: str):
        """
        Returns the appropriate activation layer based on the layer name.

        Args:
        -----
        layer_name: Name of the activation function (e.g., 'ReLU', 'ExU').
        layer_type: Type of the layer ('first' or 'hidden').

        Returns:
        --------
        activation_layer: Corresponding activation layer class.
        """
        if layer_name == 'ReLU':
            return ReLULayer
        elif layer_name == 'ExU':
            return ExULayer
        else:
            raise NotImplementedError(f"{layer_type.capitalize()} layer '{layer_name}' is not implemented.")


    def _initialize_layers(self):
        """
        Initializes the network layers based on the defined architecture.
        """
        # Initialize the first layer
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(self.activation_first_layer(1, self.num_units))

        # Add additional hidden layers if not shallow
        if not self.shallow:
            in_units = self.num_units
            for out_units in self.hidden_units:
                self.hidden_layers.append(self.activation_hidden_layer(in_units, out_units))
                in_units = out_units  # Update input size for the next layer

            # Initialize the output layer
            self._initialize_output_layer(in_units)
        
        else:
            self._initialize_output_layer(self.num_units)


    def _initialize_output_layer(self, in_units: int):
        """
        Initializes the output layer based on whether it's a monotonic network.
        
        Args:
        -----
        in_units: Number of input units for the output layer.
        """
        self.output_layer = torch.nn.Linear(in_units, self.num_classes, bias=False)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        #torch.nn.init.constant_(self.output_layer.bias, 0.01)#torch.nn.init.zeros_(self.output_layer.bias)


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
                 first_layer: str = 'ReLU',
                 hidden_layer: str = 'ReLU',         
                 num_classes: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
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

        # Create the NAM network architectures
        self._initialize_NAM_block_architecture()
    
    def _initialize_NAM_block_architecture(self):
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
            
            self.multi_output_layer = torch.nn.Linear(1, self.num_classes, bias=False)
            torch.nn.init.xavier_uniform_(self.multi_output_layer.weight)
        
        else:
            raise NotImplementedError(f"{self.architecture_type} architecture type is not implemented.")


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
            outputs = self.multi_output_layer(single_output)
                    
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
                 first_layer: str = 'ReLU',
                 hidden_layer: str = 'ReLU',        
                 num_classes: int = 1,
                 architecture_type: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 feature_to_concept_mask: torch.Tensor = None, 
                 ):
        super().__init__()

        """
        Initializes NAM Parameters.

        Perparameters:
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
        
        # Validate the feature-to-concept mask, or initialize it as a matrix of ones if not provided
        if feature_to_concept_mask is None:
            #self.feature_to_concept_mask = torch.ones(self.input_size, self.num_classes, requires_grad=False)
            self.feature_to_concept_mask = None
        else:
            assert feature_to_concept_mask.shape == (self.input_size, self.num_classes), \
                "feature_to_concept_mask should have the shape (num_features, num_concepts)"
            self.feature_to_concept_mask = feature_to_concept_mask.detach().clone().requires_grad_(False)
    
        self.feature_nns = torch.nn.ModuleList([
                                    FeatureNN(num_units=self.num_units[i],
                                            hidden_units = self.hidden_units,
                                            dropout = self.hidden_dropout,
                                            shallow = self.shallow,
                                            first_layer = self.activation_first_layer,
                                            hidden_layer = self.activation_hidden_layer,          
                                            num_classes = self.num_classes,
                                            architecture_type = self.architecture_type,
                                            ) 
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

        # -------------------------- Apply feature-to-concept mask --------------------------
        if self.feature_to_concept_mask is not None:
            masked_features = torch.matmul(f_out, self.feature_to_concept_mask.to(x.device))  # Map features to concepts
            feature_outputs = masked_features.diagonal(dim1=-2, dim2=-1) # Extract diagonal: [batch_size, num_concepts]
        else:
            # If no mask is provided, directly sum over the feature dimension
            feature_outputs = f_out.sum(axis=-1)  # Shape: [batch_size, num_concepts]
        # -----------------------------------------------------------------------------------
        outputs = feature_outputs + self.bias #if I want positive bias: F.relu(self.bias)
        
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
                 task_type: str = 'regression',
                 hierarch_net: bool = True,
                 learn_only_feature_to_concept: bool = False,
                 learn_only_concept_to_target: bool = False, 
                 #phase1 - latent_features:
                 num_units_phase1: int= 64,
                 hidden_units_phase1: list = [64, 32],
                 hidden_dropout_phase1: float = 0.,
                 feature_dropout_phase1: float = 0.,
                 shallow_phase1: bool = True,
                 first_layer_phase1: str = 'ReLU',
                 hidden_layer_phase1: str = 'ReLU',         
                 latent_var_dim: int = 1,
                 featureNN_architecture_phase1: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 feature_to_concept_mask: torch.Tensor = None, 
                 #phase2 - final outputs:
                 num_units_phase2: int= 64,
                 hidden_units_phase2: list = [64, 32],
                 hidden_dropout_phase2: float = 0.,
                 feature_dropout_phase2: float = 0.,
                 shallow_phase2: bool = True,
                 first_layer_phase2: str = 'ReLU',
                 hidden_layer_phase2: str = 'ReLU',          
                 output_dim: int = 1,
                 featureNN_architecture_phase2: str = 'multi_output',  # options: 'single_to_multi_output', 'parallel_single_output', 'multi_output', 'monotonic_hidden_layer'
                 ):
        super().__init__()

        """
        Initializes NAM Parameters.

        Parameters:
          1) num_inputs: Number of feature inputs in input data.
          2) task_type: The type of task. Options: 'regression', 'binary_classification', 'multi_classification'.
          3) hierarch_net: Boolean var that deteramine if using the hirarchial network

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
        self.latent_var_dim = latent_var_dim
        self.num_classes = output_dim
        
        self.task_type = task_type
        self.hierarch_net = hierarch_net
        self.learn_only_feature_to_concept = learn_only_feature_to_concept
        self.learn_only_concept_to_target = learn_only_concept_to_target

        if not learn_only_concept_to_target:
            # phase1 hyperparameters - concepts:
            self.num_units_phase1 = num_units_phase1
            self.hidden_units_phase1 = hidden_units_phase1
            self.hidden_dropout_phase1 = hidden_dropout_phase1
            self.feature_dropout_phase1 = feature_dropout_phase1
            self.shallow_phase1 = shallow_phase1
            self.activation_first_layer_phase1 = first_layer_phase1
            self.activation_hidden_layer_phase1 = hidden_layer_phase1
            self.architecture_type_phase1 = featureNN_architecture_phase1
            self.feature_to_concept_mask = feature_to_concept_mask

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
                            feature_to_concept_mask = self.feature_to_concept_mask,         
                            )

        if hierarch_net or not learn_only_feature_to_concept:
            # phase2 hyperparameters - final outputs:
            self.num_units_phase2 = num_units_phase2
            self.hidden_units_phase2 = hidden_units_phase2
            self.hidden_dropout_phase2 = hidden_dropout_phase2
            self.feature_dropout_phase2 = feature_dropout_phase2
            self.shallow_phase2 = shallow_phase2
            self.activation_first_layer_phase2 = first_layer_phase2
            self.activation_hidden_layer_phase2 = hidden_layer_phase2
            self.architecture_type_phase2 = featureNN_architecture_phase2

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
                                    feature_to_concept_mask = None,   
                                    )
        
        # Define activation based on task type
        if self.task_type == 'binary_classification':
            self.final_activation = torch.nn.Sigmoid()
        elif self.task_type == 'multi_classification':
            self.final_activation = torch.nn.Softmax(dim=-1)  # Softmax for multi-class classification
        else:
            self.final_activation = None  # No activation for regression (linear output)


    def forward(self, x):
        if self.learn_only_concept_to_target:
            _, concepts, phase1_gams_out, _ = SyntheticDatasetGenerator.get_synthetic_data_phase1(num_exp=x.size(0), raw_features=self.input_size, num_concepts=self.latent_var_dim, is_test=False, seed=0, X_val=x)
            outputs, phase2_gams_out = self.NAM_output(concepts)
            # Apply activation based on the task
            if self.final_activation:
                outputs = self.final_activation(outputs)
            return outputs, concepts, phase1_gams_out, phase2_gams_out
        
        else:
            concepts, phase1_gams_out = self.NAM_features(x)

            if self.learn_only_feature_to_concept: # Phase 2 computation using latent features
                outputs, phase2_gams_out = SyntheticDatasetGenerator.get_synthetic_data_phase2(concepts, num_classes=self.num_classes, is_test=False)
            else:          
                outputs, phase2_gams_out = self.NAM_output(concepts)
            # Apply activation based on the task
            if self.final_activation:
                outputs = self.final_activation(outputs)

            return outputs, concepts, phase1_gams_out, phase2_gams_out