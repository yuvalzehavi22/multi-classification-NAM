import argparse
from model.activation_layers import ExULayer, ReLULayer, LipschitzMonotonicLayer
import sys

def parse_args():

    parser = argparse.ArgumentParser(description="Define experiment configuration")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument(
        '--WB_project_name', 
        type=str, 
        default="Hirarchial GAMs", 
        help="options:'Hirarchial_GAMs-synt_data', 'GAMs-synt_data_phase1', 'GAMs-synt_data_phase2', 'Hirarchial_GAMs-classification', 'Hirarchical_NAMs_hyperparam_optimization'")
    
    # ---------------------------------------------------------------
    # ----------------------- Data parameters -----------------------
    # ---------------------------------------------------------------
    parser.add_argument('--num_exp', type=int, default=1000, help='Number of examples - for the synthetic data')
    parser.add_argument('--in_features', type=int, default=5, help='Number of raw features')
    parser.add_argument('--latent_dim', type=int, default=3, help='Number of context features (The output of phase1)')
    parser.add_argument('--output_dim', type=int, default=4, help='Number of output classes')
    parser.add_argument('--val_split', type=float, default=0.2, help='The percentage of the data that will make up the validation set')
    
    # ----------------------------------------------------------------------------------
    # ----------------------- model parameters (for both phases) -----------------------
    # ----------------------------------------------------------------------------------
    # Model architecture
    parser.add_argument('--task_type', type=str, default='regression', help="The type of task. Options: 'regression', 'binary_classification', 'multi_classification'")
    parser.add_argument(
        '--hierarch_net', 
        type=int,
        choices=[0, 1],  # Only allow 0 or 1 as valid input
        default=1,       # Default value: 0 for False 
        help="Use hierarchical net (adding phase2) - 0 for False, 1 for True"
    )
    parser.add_argument(
        '--learn_only_feature_to_concept', 
        type=int,
        choices=[0, 1],  # Only allow 0 or 1 as valid input
        default=0,       # Default value: 0 for False 
        help="Learn only concepts (the shape functions for phase2 are geven) - 0 for False, 1 for True"
    )
    parser.add_argument(
        '--learn_only_concept_to_target', 
        type=int,
        choices=[0, 1],  # Only allow 0 or 1 as valid input
        default=0,       # Default value: 0 for False 
        help="Learn only concept_to_target phase (the shape functions for phase1 are geven) - 0 for False, 1 for True"
    )
    parser.add_argument(
        '--use_feature_concept_mask', 
        type=int,
        choices=[0, 1],  # Only allow 0 or 1 as valid input
        default=0,       # Default value: 0 for False 
        help=" Using a binary matrix that defines for each feature which concepts it contributes to (0 for False, 1 for True)"
    )

    parser.add_argument(
        "--featureNN_arch_phase1", 
        type=str, 
        default="single_to_multi_output", 
        help="one of 'multi_output', 'single_to_multi_output' or 'parallel_single_output'")
    parser.add_argument(
        "--featureNN_arch_phase2", 
        type=str, 
        default="parallel_single_output", 
        help="one of 'multi_output', 'single_to_multi_output' or 'parallel_single_output'")
    # Networks parameter
    parser.add_argument(
        '--first_activate_layer_phase1', 
        type=str, 
        default='ReLU', 
        help='First activation layer for phase1. options: ReLU, ExU, LipschitzMonotonic'
    )
    parser.add_argument(
        '--first_activate_layer_phase2', 
        type=str, 
        default='ReLU', 
        help='First activation layer for phase2 (optional). options: ReLU, ExU, LipschitzMonotonic'
    )
    parser.add_argument(
        '--first_hidden_dim_phase1', 
        type=int, 
        default=64, 
        help='Number of hidden units in the first hidden layer for phase1'
    )
    parser.add_argument(
        '--first_hidden_dim_phase2', 
        type=int, 
        default=64, 
        help='Number of hidden units in the first hidden layer for phase2'
    )
    parser.add_argument(
        '--shallow_phase1', 
        type=int, 
        choices=[0, 1],
        default=0, 
        help="0 for False, 1 for True. If True, then a shallow network with a single hidden layer is created - the model will not use the 'hidden_activate_layer_phase1'"
    )
    parser.add_argument(
        '--shallow_phase2', 
        type=int,
        choices=[0, 1],
        default=0,
        help="0 for False, 1 for True. If True, then a shallow network with a single hidden layer is created - the model will not use the 'hidden_activate_layer_phase2'"
    )
    # If not shallow - adding more layers to the network
    parser.add_argument(
        '--hidden_activate_layer_phase1', 
        type=str, 
        default='ReLU', 
        help='Hidden activation layer for phase1. options: ReLU, ExU, LipschitzMonotonic'
    )
    parser.add_argument(
        '--hidden_activate_layer_phase2', 
        type=str, 
        default='ReLU', 
        help='Hidden activation layer for phase2 (optional). options: ReLU, ExU, LipschitzMonotonic'
    )
    parser.add_argument(
        '--hidden_dim_phase1', 
        type=int, 
        nargs='+',
        default=[64, 32],
        help='Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.'
    )
    parser.add_argument(
        '--hidden_dim_phase2', 
        type=int, 
        nargs='+',
        default=[64, 32],
        help='Number of hidden units in the next hidden layers - the number of additional layers is the size of the list.'
    )
    parser.add_argument(
        '--hidden_dropout_phase1', 
        type=float, 
        default=0.0, 
        help='Coefficient for dropout within each Feature NNs'
    )
    parser.add_argument(
        '--hidden_dropout_phase2', 
        type=float, 
        default=0.0, 
        help='Coefficient for dropout within each Feature NNs'
    )
    parser.add_argument(
        '--feature_dropout_phase1', 
        type=float, 
        default=0.0, #0.3
        help='Coefficient for dropping out entire Feature NNs'
    )
    parser.add_argument(
        '--feature_dropout_phase2', 
        type=float, 
        default=0.0, 
        help='Coefficient for dropping out entire Feature NNs'
    )
    # Monotonic parameters
    parser.add_argument('--weight_norms_kind_phase1', type=str, default='one-inf', help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--weight_norms_kind_phase2', type=str, default='one-inf', help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--group_size_phase1', type=int, default=2, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--group_size_phase2', type=int, default=2, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument(
        '--monotonic_constraint_phase1', 
        type=int, 
        nargs='+',
        default=None, 
        help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument(
        '--monotonic_constraint_phase2', 
        type=int, 
        nargs='+',
        default=None,
        help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')

    # ---------------------------------------------------------------------
    # ----------------------- train/eval parameters -----------------------
    # ---------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0035, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--clip_value', type=int, default=0, help='Clip value of the wigths')
    # regularization parameters
    parser.add_argument("--l2_lambda_phase1",type=float, default=0.0, help="l2 regularization for the gams outputs of phase1")
    parser.add_argument("--l2_lambda_phase2",type=float, default=0.0, help="l2 regularization for the gams outputs of phase2")
    parser.add_argument("--l1_lambda_phase1",type=float, default=0.0, help="l1 regularization for the gams outputs of phase1")
    parser.add_argument("--l1_lambda_phase2",type=float, default=0.0, help="l1 regularization for the gams outputs of phase2")
    parser.add_argument("--monotonicity_lambda_phase1",type=float, default=0.0, help="Parameter to controls the strength of the monotonicity constraint - Phase1")
    parser.add_argument("--monotonicity_lambda_phase2",type=float, default=0.0, help="Parameter to controls the strength of the monotonicity constraint - Phase2")

    parser.add_argument('--eval_every', type=int, default=20, help='Evaluate every N epochs')
    parser.add_argument('--early_stop_delta', type=float, default=0.0, help='Min delta for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=200, help='Patience for early stopping')
    # parser.add_argument(
    #     "--monitor",
    #     type=str,
    #     default="val_loss",
    #     help="(val_)loss or (val_)metric name to monitor",
    # )
    parser.add_argument(
        '--track_gradients', 
        type=int,
        choices=[0, 1],  # Only allow 0 or 1 as valid input
        default=0,       # Default value: 0 for False 
        help="track_gradients - 0 for False, 1 for True"
    )
    # --------------------------------------------------------------------
    # ----------------------- Optimizer parameters -----------------------
    # --------------------------------------------------------------------
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Only Adam, AdamW, and RAdam are considered",
    )

    # Scheduler parameters
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="CosineAnnealingLR",
        help="one of 'StepLR', 'ReduceLROnPlateau', 'CyclicLR' or 'OneCycleLR', 'CosineAnnealingLR', NoScheduler",
    )
    # StepLR params
    parser.add_argument(
        "--StepLR_step_size",
        type=int,
        default=10,
        help="Period of learning rate decay",
    )
    parser.add_argument(
        "--StepLR_gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay",
    )
    # ReduceLROnPlateau (rop) params
    parser.add_argument(
        "--rop_mode",
        type=str,
        default="min",
        help="One of min, max",
    )
    parser.add_argument(
        "--rop_factor",
        type=float,
        default=0.2,
        help="Factor by which the learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_threshold",
        type=float,
        default=0.001,
        help="Threshold for measuring the new optimum",
    )
    parser.add_argument(
        "--rop_threshold_mode",
        type=str,
        default="abs",
        help="One of rel, abs",
    )
    parser.add_argument(
        "--rop_min_lr",
        type=float,
        default=0,
        help="min lr",
    )
    # OneCycleLR (oclr) params
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.01,
        help="max_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="the total number of steps in the cycle",
    )
    parser.add_argument(
        "--oclr_epochs",
        type=int,
        default=10,
        help="the number of epochs to train for",
    )
    parser.add_argument(
        "--oclr_steps_per_epoch",
        type=int,
        default=100,
        help="the number of steps per epoch to train for",
    )
    parser.add_argument(
        "--div_factor",
        type=float,
        default=25,
        help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=1e4,
        help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
    )
    parser.add_argument(
        "--cycle_momentum",
        action="store_true",
    )
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.3,
        help="Percentage of the cycle (in number of steps) spent increasing the learning rate",
    )
    parser.add_argument(
        "--anneal_strategy",
        type=str,
        default='cos',
        help="{'cos', 'linear'} Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing",
    )  
    # CyclicLR 
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,
        help="base_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--step_size_up",
        type=int,
        default=2000,
        help="Number of training iterations in the increasing half of a cycle",
    )
    parser.add_argument(
        "--step_size_down",
        type=int,
        default=None,
        help="Number of training iterations in the decreasing half of a cycle",
    )

    parser.add_argument("--trial_id", type=str, help="Parameter for hyperparameter tunning")

    # ----------------------- save parameters -----------------------
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    args, unknown = parser.parse_known_args()
    return args