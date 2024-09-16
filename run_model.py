import numpy as np
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
import math
import os
import random
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import wandb

from data_processing.data_loader import *
from model.activation_layers import ExULayer, ReLULayer, LipschitzMonotonicLayer
from model.model_network import HierarchNeuralAdditiveModel
from utils.visualize_shape_functions import get_shape_functions, get_shape_functions_synthetic_data
from utils.model_architecture_type import get_defult_architecture_phase1, get_defult_architecture_phase2
from training.trainer import Trainer
from training.trainer_utils import visualize_loss
from utils.utils import define_device, seed_everything
from utils.model_parser import parse_args



def main():
    # Parsing arguments
    args = parse_args()

    # Initialize W&B run and log the parameters
    wandb.init(project="Hirarchial GAMs", config=args)
    
    # Set device and seed
    device = define_device("auto")
    print(device)
    seed_everything(args.seed)

    # DATA PROCESSING: Generate synthetic data for Phase 1 and Phase 2
    X, y_phase1, _ = SyntheticDatasetGenerator.get_synthetic_data_phase1(args.num_exp, args.in_features)
    y_phase2, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1)

    SyntheticDataset= True
    if SyntheticDataset:
        # Generate synthetic data for validation set
        X_val, y_phase1_val, _ = SyntheticDatasetGenerator.get_synthetic_data_phase1(10000, args.in_features)
        y_phase2_val, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1_val)

        train_loader = SyntheticDatasetGenerator.make_loader(X, y_phase2, batch_size=args.batch_size)
        val_loader = SyntheticDatasetGenerator.make_loader(X_val, y_phase2_val, batch_size=args.batch_size)
    else:
        # Initialize DataLoaderWrapper with validation split
        dataloader_wrapper = DataLoaderWrapper(X, y_phase2, val_split=args.val_split)
        # Create DataLoaders
        train_loader, val_loader = dataloader_wrapper.create_dataloaders()

    
    print("Train size:", len(train_loader.dataset), "Val size:", len(val_loader.dataset))

    # # Get default architectures for both phases - block_layers_type options: ReLU, shallow_ExU, Monotonic, ExU_ReLU
    # get_defult_architecture_phase1(args, block_layers_type=args.GAM_block_layers_type_phase1)
    # get_defult_architecture_phase2(args, block_layers_type=args.GAM_block_layers_type_phase2)

    print("Training Hierarchical NAM...")
    print(f"Phase1 architecture: [{args.first_activate_layer_phase1}: {args.first_hidden_dim_phase1}, {args.hidden_activate_layer_phase1}: {args.hidden_dim_phase1}]")
    print(f"Phase2 architecture: [{args.first_activate_layer_phase2}: {args.first_hidden_dim_phase2}, {args.hidden_activate_layer_phase2}: {args.hidden_dim_phase2}]")

    # Model definition: HierarchNeuralAdditiveModel
    hirarch_nam = HierarchNeuralAdditiveModel(num_inputs=args.in_features,
                                        task_type= args.task_type,
                                        hierarch_net= args.hierarch_net,
                                        #phase1 - latent_features:
                                        num_units_phase1= args.first_hidden_dim_phase1,
                                        hidden_units_phase1 = args.hidden_dim_phase1,
                                        hidden_dropout_phase1 =args.hidden_dropout_phase1,
                                        feature_dropout_phase1 = args.feature_dropout_phase1,
                                        shallow_phase1 = args.shallow_phase1,
                                        first_layer_phase1 = args.first_activate_layer_phase1,
                                        hidden_layer_phase1= args.hidden_activate_layer_phase1,         
                                        latent_var_dim= args.latent_dim,
                                        featureNN_architecture_phase1= args.featureNN_arch_phase1,
                                        weight_norms_kind_phase1= args.weight_norms_kind_phase1, 
                                        group_size_phase1= args.group_size_phase1, 
                                        monotonic_constraint_phase1= args.monotonic_constraint_phase1,
                                        #phase2 - final outputs:
                                        num_units_phase2= args.first_hidden_dim_phase2,
                                        hidden_units_phase2 = args.hidden_dim_phase2,
                                        hidden_dropout_phase2 = args.hidden_dropout_phase2,
                                        feature_dropout_phase2 = args.feature_dropout_phase1,
                                        shallow_phase2 = args.shallow_phase2,
                                        first_layer_phase2 = args.first_activate_layer_phase2,
                                        hidden_layer_phase2 = args.hidden_activate_layer_phase2,          
                                        output_dim = args.output_dim,
                                        featureNN_architecture_phase2 = args.featureNN_arch_phase2,
                                        weight_norms_kind_phase2 = args.weight_norms_kind_phase2, 
                                        group_size_phase2 = args.group_size_phase2, 
                                        monotonic_constraint_phase2 = args.monotonic_constraint_phase2
                                        ).to(device)

    # Initialize the Trainer class
    trainer = Trainer(
        model=hirarch_nam,
        optimizer=args.optimizer,
        loss_function=None,
        lr_scheduler=args.lr_scheduler, 
        scheduler_params=None,
        eval_metric=None, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        l1_lambda_phase1=args.l1_lambda_phase1,
        l1_lambda_phase2=args.l1_lambda_phase2,
        l2_lambda_phase1=args.l2_lambda_phase1,
        l2_lambda_phase2=args.l2_lambda_phase2,
        monotonicity_lambda=args.monotonicity_lambda,
        eval_every=args.eval_every,
        early_stop_delta=args.early_stop_delta,
        early_stop_patience=args.early_stop_patience,
        clip_value=args.clip_value,
        device_name="auto"
    )

    # Run the training phase
    train_loss_history, val_loss_history = trainer.train(args, train_loader, val_loader)

    # print loss curves
    if 0:
        visualize_loss(train_loss_history, val_loss_history)
    
    # Visualization of the shape functions created in the two phases
    if SyntheticDataset:
        get_shape_functions_synthetic_data(hirarch_nam, args, num_test_exp=500)
    else: 
        get_shape_functions(hirarch_nam, args)


if __name__ == "__main__":
    main()
                        


#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0035 --epochs 1000 --l1_lambda_phase1 0.001
# --optimizer "Adam" --epochs 100 --batch_size 1024 --learning_rate 0.0035 --weight_decay 0.0001 --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 128 --hidden_dim_phase2 128 64 --first_activate_layer_phase2 "LipschitzMonotonic" --hidden_activate_layer_phase2 "LipschitzMonotonic"
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-5 --l1_lambda_phase2 1e-6
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6 --first_hidden_dim_phase2 64 --hidden_dim_phase2 64 32 --first_activate_layer_phase2 "ReLU" --hidden_activate_layer_phase2 "ReLU"