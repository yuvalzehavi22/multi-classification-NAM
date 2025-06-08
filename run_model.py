import json
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
# from data_processing.data_loader_new import *

from model.activation_layers import ExULayer, ReLULayer, LipschitzMonotonicLayer
from model.model_network import HierarchNeuralAdditiveModel
from training.trainer_learn_concepts import Trainer_concepts
from utils.visualize_shape_functions import get_shape_functions, get_shape_functions_synthetic_data
from utils.model_architecture_type import get_defult_architecture_phase1, get_defult_architecture_phase2
from training.trainer import Trainer
from training.trainer_utils import get_param_groups, set_lr_scheduler_params, visualize_loss
from utils.utils import define_device, plot_concepts_weights, plot_data_histograms, plot_pred_data_histograms, seed_everything
from utils.model_parser import parse_args


def main():
    # Parsing arguments
    args = parse_args()

    # Initialize W&B run and log the parameters
    wandb.init(project=args.WB_project_name, config=args)
    
    # Set device and seed
    device = define_device("auto")
    print(f"Using device: {device}")
    seed_everything(args.seed)

    # DATA PROCESSING: Generate synthetic data for Phase 1 and Phase 2
    X, y_phase1, _, out_weights = SyntheticDatasetGenerator.get_synthetic_data_phase1(num_exp=args.num_exp, raw_features=args.in_features, num_concepts=args.latent_dim, is_test=False, seed=args.seed)
    y_phase2, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1, num_classes=args.output_dim, is_test=False)
    print("Raw features shape:", X.shape)
    print("Concepts shape:", y_phase1.shape)
    print("Outputs shape:", y_phase2.shape)

    #---------------------------------------------------------------------------------------
    # -------------------------- creating feature to concept mask --------------------------
    if args.use_feature_concept_mask:
        feature_to_concept_mask = torch.zeros(args.in_features, args.latent_dim, device=device)

        # Convert out_weights dictionary keys into a tensor mask
        for key, value in out_weights.items():
            if value != 0:
                output_idx, feature_idx = map(int, key.split('_')[1:])  # Extract indices from the key
                feature_to_concept_mask[feature_idx, output_idx] = 1
    else:
        feature_to_concept_mask = None
    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------

    SyntheticDataset= True
    if SyntheticDataset:
        # Generate synthetic data for validation set
        # X_val, y_phase1_val, _, _ = SyntheticDatasetGenerator.get_synthetic_data_phase1(100, args.in_features)
        # y_phase2_val, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1_val)

        if args.hierarch_net:
            train_loader = SyntheticDatasetGenerator.make_loader(X, y_phase2, batch_size=args.batch_size)
            #val_loader = SyntheticDatasetGenerator.make_loader(X_val, y_phase2_val, batch_size=args.batch_size)
        else:
            train_loader = SyntheticDatasetGenerator.make_loader(X, y_phase1, batch_size=args.batch_size)
            #val_loader = SyntheticDatasetGenerator.make_loader(X_val, y_phase1_val, batch_size=args.batch_size)
    else:
        # Initialize DataLoaderWrapper with validation split
        dataloader_wrapper = DataLoaderWrapper(X, y_phase2, val_split=args.val_split)
        # Create DataLoaders
        train_loader, val_loader = dataloader_wrapper.create_dataloaders()

    # Save the data distribution plots
    input_fig = plot_data_histograms(values=X, values_name='Input', save_path="data_processing/plots/")
    #input_fig.show()
    concept_fig = plot_data_histograms(values=y_phase1, values_name='Concept',nbins=80, model_predict=False, save_path="data_processing/plots/")
    #concept_fig.show()
    target_fig = plot_data_histograms(values=y_phase2, values_name='Target',nbins=100, model_predict=False, save_path="data_processing/plots/")
    #target_fig.show()

    print("Training Hierarchical NAM...")

    print(f"Phase1 architecture: [{args.first_activate_layer_phase1}: {args.first_hidden_dim_phase1}, {args.hidden_activate_layer_phase1}: {args.hidden_dim_phase1}]")
    if args.hierarch_net:
        print(f"Phase2 architecture: [{args.first_activate_layer_phase2}: {args.first_hidden_dim_phase2}, {args.hidden_activate_layer_phase2}: {args.hidden_dim_phase2}]")

    # Model definition: HierarchNeuralAdditiveModel
    hirarch_nam = HierarchNeuralAdditiveModel(num_inputs=args.in_features,
                                        task_type= args.task_type,
                                        hierarch_net= args.hierarch_net,
                                        learn_only_feature_to_concept = args.learn_only_feature_to_concept,
                                        learn_only_concept_to_target = args.learn_only_concept_to_target,
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
                                        feature_to_concept_mask= feature_to_concept_mask,
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
                                        ).to(device)
    
    total_params = sum(p.numel() for p in hirarch_nam.parameters())
    print(f"Number of parameters: {total_params}")
    
    # # Watch model weights and gradients
    # wandb.watch(hirarch_nam, log="gradients", log_freq=args.batch_size)

    # if args.featureNN_arch_phase1 == 'single_to_multi_output':
    #     plot_concepts_weights(out_weights, hirarch_nam, model_predict = False)
    
    scheduler_params = set_lr_scheduler_params(args, args.lr_scheduler)
    print(scheduler_params)

    # Initialize the Trainer class (Trainer_concepts or Trainer)
    trainer = Trainer(
        model=hirarch_nam,
        optimizer=args.optimizer,
        loss_function=None,
        lr_scheduler=args.lr_scheduler, 
        scheduler_params=scheduler_params,
        eval_metric=None, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        l1_lambda_phase1=args.l1_lambda_phase1,
        l1_lambda_phase2=args.l1_lambda_phase2,
        l2_lambda_phase1=args.l2_lambda_phase1,
        l2_lambda_phase2=args.l2_lambda_phase2,
        monotonicity_lambda_phase1=args.monotonicity_lambda_phase1,
        monotonicity_lambda_phase2=args.monotonicity_lambda_phase2,
        eval_every=args.eval_every,
        early_stop_delta=args.early_stop_delta,
        early_stop_patience=args.early_stop_patience,
        clip_value=args.clip_value,
        device_name="auto"
    )

    if args.track_gradients:
        all_param_groups = get_param_groups(hirarch_nam, args)
    else:
        all_param_groups=None
        
    # Run the training phase
    train_loss_history, val_loss_history = trainer.train(loader=train_loader, all_param_groups=all_param_groups)
    #train_loss_history, val_loss_history = trainer.train(loader=train_loader, all_param_groups=all_param_groups, val_loader=val_loader)

    # # For hyperparam tunningy
    # val_loss_data = {"val_loss": val_loss_history[-1]}
    # with open(f"val_loss_trial_{args.trial_id}.json", "w") as f:
    #     json.dump(val_loss_data, f)

    # print loss curves
    if 0:
        visualize_loss(train_loss_history, val_loss_history)

    # load the model
    hirarch_nam.load_state_dict(torch.load('/home/yuvalzehavi1/Repos/multi-classification-NAM/best_model.pt'))
    
    # Visualization of the shape functions created in the two phases
    if SyntheticDataset:
        get_shape_functions_synthetic_data(hirarch_nam, args, num_test_exp=1000)
    else: 
        get_shape_functions(hirarch_nam, args)

    # Log the predicted output distribution to W&B
    plot_pred_data_histograms(hirarch_nam, args.hierarch_net, X)

    # Log the concepts weights to W&B
    if args.featureNN_arch_phase1 == 'single_to_multi_output' and not args.learn_only_concept_to_target:
        plot_concepts_weights(out_weights, hirarch_nam, model_predict = True)
    
    
if __name__ == "__main__":
    main()
                        