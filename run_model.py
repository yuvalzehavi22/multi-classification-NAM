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
    print(device)
    seed_everything(args.seed)

    # DATA PROCESSING: Generate synthetic data for Phase 1 and Phase 2
    # X, y_phase1, _, out_weights = SyntheticDatasetGenerator.get_synthetic_data_phase1(args.num_exp, args.in_features)
    # y_phase2, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1)

    X, y_phase1, _, out_weights = SyntheticDatasetGenerator.get_synthetic_data_phase1(num_exp=args.num_exp, raw_features=args.in_features, num_concepts=args.latent_dim, is_test=False, seed=args.seed)
    y_phase2, _ = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1, num_classes=args.output_dim, is_test=False)

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
    
    #print("Train size:", len(train_loader.dataset), "Val size:", len(val_loader.dataset))


    print("Training Hierarchical NAM...")
    print(f"Phase1 architecture: [{args.first_activate_layer_phase1}: {args.first_hidden_dim_phase1}, {args.hidden_activate_layer_phase1}: {args.hidden_dim_phase1}]")
    if args.hierarch_net:
        print(f"Phase2 architecture: [{args.first_activate_layer_phase2}: {args.first_hidden_dim_phase2}, {args.hidden_activate_layer_phase2}: {args.hidden_dim_phase2}]")

    # Model definition: HierarchNeuralAdditiveModel
    hirarch_nam = HierarchNeuralAdditiveModel(num_inputs=args.in_features,
                                        task_type= args.task_type,
                                        hierarch_net= args.hierarch_net,
                                        learn_only_concepts = args.learn_only_concepts,
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
    
    total_params = sum(p.numel() for p in hirarch_nam.parameters())
    print(f"Number of parameters: {total_params}")
    
    # # Watch model weights and gradients
    # wandb.watch(hirarch_nam, log="gradients", log_freq=args.batch_size)

    # if args.featureNN_arch_phase1 == 'single_to_multi_output':
    #     plot_concepts_weights(out_weights, hirarch_nam, model_predict = False)
    
    scheduler_params = set_lr_scheduler_params(args, args.lr_scheduler)
    print(scheduler_params)

    # Initialize the Trainer class
    #Trainer_concepts or Trainer
    trainer = Trainer_concepts(
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
        get_shape_functions_synthetic_data(hirarch_nam, args, num_test_exp=500)
    else: 
        get_shape_functions(hirarch_nam, args)

    # Log the predicted output distribution to W&B
    plot_pred_data_histograms(hirarch_nam, args.hierarch_net, X)

    # Log the concepts weights to W&B
    if args.featureNN_arch_phase1 == 'single_to_multi_output':
        plot_concepts_weights(out_weights, hirarch_nam, model_predict = True)
    
    
if __name__ == "__main__":
    main()
                        

#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0035 --epochs 1000 --l1_lambda_phase1 0.001
#python run_model.py --optimizer "Adam" --epochs 100 --batch_size 1024 --learning_rate 0.0035 --weight_decay 0.0001 --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 128 --hidden_dim_phase2 128 64 --first_activate_layer_phase2 "LipschitzMonotonic" --hidden_activate_layer_phase2 "LipschitzMonotonic"
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-5 --l1_lambda_phase2 1e-6
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6
#python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0005 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6 --first_hidden_dim_phase2 64 --hidden_dim_phase2 64 32 --first_activate_layer_phase2 "ReLU" --hidden_activate_layer_phase2 "ReLU"
#python run_model.py --seed 42 --eval_every 50 --learning_rate 0.001 --epochs 1000 --hierarch_net 0 --featureNN_arch_phase1 'single_to_multi_output' --batch_size 32 --lr_scheduler 'StepLR' --l2_lambda_phase1 1e-6
#python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --num_exp 10 --epochs 2 --lr_scheduler 'CosineAnnealingLR' --learning_rate 0.0005 --in_features 5 --latent_dim 3 --output_dim 4 --learn_only_concepts 1