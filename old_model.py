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
    


@dataclass
class TabModel(BaseEstimator):
    """ Class for TabNet model."""
    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    cat_emb_dim: int = 1
    n_independent: int = 2
    n_shared: int = 2
    epsilon: float = 1e-15
    momentum: float = 0.02
    lambda_sparse: float = 1e-3
    seed: int = 0
    clip_value: int = 1
    verbose: int = 1
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2))
    scheduler_fn: Any = None
    scheduler_params: Dict = field(default_factory=dict)
    mask_type: str = "sparsemax"
    input_dim: int = None
    output_dim: int = None
    device_name: str = "auto"
    n_shared_decoder: int = 1
    n_indep_decoder: int = 1
    grouped_features: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        # These are default values needed for saving model
        self.batch_size = 1024
        self.virtual_batch_size = 128

        torch.manual_seed(self.seed)
        # Defining device
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            warnings.warn(f"Device used : {self.device}")

        # create deep copies of mutable parameters
        self.optimizer_fn = copy.deepcopy(self.optimizer_fn)
        self.scheduler_fn = copy.deepcopy(self.scheduler_fn)

        updated_params = check_embedding_parameters(self.cat_dims,
                                                    self.cat_idxs,
                                                    self.cat_emb_dim)
        self.cat_dims, self.cat_idxs, self.cat_emb_dim = updated_params


    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
            dict for custom weights per class
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_input(X_train)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        if self.compute_importance:
            # compute feature importance once the best model is defined
            self.feature_importances_ = self._compute_feature_importances(X_train)

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)


    def save_model(self, path):
        """Saving TabNet model in two distinct files.

        Parameters
        ----------
        path : str
            Path of the model.

        Returns
        -------
        str
            input filepath with ".zip" appended

        """
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath):
        """Load TabNet model.

        Parameters
        ----------
        filepath : str
            Path of the model.
        """
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params["init_params"])

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        self.load_class_attrs(loaded_params["class_attrs"])

        return

    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()

        for batch_idx, (X, y) in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)

            batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        if self.augmentations is not None:
            X, y = self.augmentations(X, y)

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()

        # compute model output
        scores, _ = self.network(X)

        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()

        return scores

    def _set_network(self):
        """Setup the network ."""
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)


    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=(
                    self._metrics[-1]._maximize if len(self._metrics) > 0 else None
                ),
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            wrn_msg = "No early stopping will be performed, last training weights will be used."
            warnings.warn(wrn_msg)

        if self.scheduler_fn is not None:
            # Add LR Scheduler call_back
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(
            self.network.parameters(), **self.optimizer_params
        )

    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders


    @abstractmethod
    def update_fit_params(self, X_train, y_train, eval_set, weights):
        """
        Set attributes relative to fit function.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
        """
        raise NotImplementedError(
            "users must define update_fit_params to use this base class"
        )

    @abstractmethod
    def compute_loss(self, y_score, y_true):
        """
        Compute the loss.

        Parameters
        ----------
        y_score : a :tensor: `torch.Tensor`
            Score matrix
        y_true : a :tensor: `torch.Tensor`
            Target matrix

        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError(
            "users must define compute_loss to use this base class"
        )

    @abstractmethod
    def prepare_target(self, y):
        """
        Prepare target before training.

        Parameters
        ----------
        y : a :tensor: `torch.Tensor`
            Target matrix.

        Returns
        -------
        `torch.Tensor`
            Converted target matrix.
        """
        raise NotImplementedError(
            "users must define prepare_target to use this base class"
        )