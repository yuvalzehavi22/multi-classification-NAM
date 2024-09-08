import argparse
from model import ExULayer, ReLULayer, LipschitzMonotonicLayer

def parse_args():

    parser = argparse.ArgumentParser(description="Define experiment configuration")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # ----------------------- Data parameters -----------------------
    parser.add_argument('--num_exp', type=int, default=50000, help='Number of experiments')
    parser.add_argument('--in_features', type=int, default=10, help='Number of input features')
    parser.add_argument('--latent_dim', type=int, default=4, help='Number of context features (The output of phase1)')
    parser.add_argument('--output_dim', type=int, default=2, help='Number of output classes')

    # ----------------------- model parameters (for both phases) -----------------------
    parser.add_argument(
        '--ActivateLayers_pase1', 
        type=str, 
        default='ReLU', 
        help='Activation layer for phase1. options: ReLU, ExU, LipschitzMonotonic, ExU_ReLU'
    )
    parser.add_argument(
        '--ActivateLayers_pase2', 
        type=str, 
        default=None, 
        help='Activation layer for phase2 (optional). options: ReLU, ExU, LipschitzMonotonic, ExU_ReLU'
    )

    parser.add_argument('--hidden_dropout_phase1', type=int, default=0, help='Coefficient for dropout within each Feature NNs')
    parser.add_argument('--hidden_dropout_phase2', type=int, default=0, help='Coefficient for dropout within each Feature NNs')

    parser.add_argument('--feature_dropout_phase1', type=int, default=0, help='Coefficient for dropping out entire Feature NNs')
    parser.add_argument('--feature_dropout_phase2', type=int, default=0, help='Coefficient for dropping out entire Feature NNs')

    parser.add_argument(
        "--featureNN_arch_phase1", 
        type=str, 
        default="multi_output", 
        help="one of 'multi_output', 'single_to_multi_output', 'parallel_single_output' or 'monotonic_hidden_layer'")
    parser.add_argument(
        "--featureNN_arch_phase2", 
        type=str, 
        default="multi_output", 
        help="one of 'multi_output', 'single_to_multi_output', 'parallel_single_output' or 'monotonic_hidden_layer'")
    # monotonic_hidden_layer
    parser.add_argument('--weight_norms_kind_phase1', type=str, default='one-inf', help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--weight_norms_kind_phase2', type=str, default='one-inf', help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--group_size_phase1', type=int, default=2, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--group_size_phase2', type=int, default=2, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--monotonic_constraint_phase1', type=int, default=None, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')
    parser.add_argument('--monotonic_constraint_phase2', type=int, default=None, help='From EXPRESSIVE MONOTONIC NEURAL NETWORKS')


    # ----------------------- train/eval parameters -----------------------
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0035, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument("--l2_regularization",type=float, default=0.0, help="l2 weight decay")

    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--early_stop_delta', type=float, default=0.0, help='Min delta for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="(val_)loss or (val_)metric name to monitor",
    )

    # ----------------------- Optimizer parameters -----------------------
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
        default="ReduceLROnPlateau",
        help="one of 'ReduceLROnPlateau', 'CyclicLR' or 'OneCycleLR', NoScheduler",
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
    # CyclicLR and OneCycleLR params
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,
        help="base_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.01,
        help="max_lr for cyclic lr_schedulers",
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
        "--n_cycles",
        type=float,
        default=5,
        help="number of cycles for CyclicLR",
    )
    parser.add_argument(
        "--cycle_momentum",
        action="store_true",
    )
    parser.add_argument(
        "--pct_step_up",
        type=float,
        default=0.3,
        help="Percentage of the cycle (in number of steps) spent increasing the learning rate",
    )

    # ----------------------- save parameters -----------------------
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    args = parser.parse_args()
    return args


def define_experiment(args):
    config = dict()

    hirarchical_net = [args.ActivateLayers_pase1]
    if args.ActivateLayers_pase2 is not None:
        hirarchical_net.append(args.ActivateLayers_pase2)
        
    config['hirarchical_net'] = hirarchical_net

    for phase in range(len(hirarchical_net)):
        if hirarchical_net[phase] == 'ReLU':
            config[f'first_ActivateLayer_phase{phase+1}'] = ReLULayer
            config[f'first_hidden_dim_phase{phase+1}'] = 64           
            config[f'shallow_phase{phase+1}'] = False
            config[f'hidden_ActivateLayer_phase{phase+1}'] = ReLULayer
            config[f'hidden_dim_phase{phase+1}'] = [64, 32]
            config[f'hidden_dropout_phase{phase+1}'] = args.hidden_dropout_phase1 if phase == 0 else args.hidden_dropout_phase2
            config[f'feature_dropout_phase{phase+1}'] = args.feature_dropout_phase1 if phase == 0 else args.feature_dropout_phase2

        elif hirarchical_net[phase] == 'ExU':
            config[f'first_ActivateLayer_phase{phase+1}'] = ExULayer
            config[f'first_hidden_dim_phase{phase+1}'] = 1024           
            config[f'shallow_phase{phase+1}'] = True
            config[f'hidden_dropout_phase{phase+1}'] = args.hidden_dropout_phase1 if phase == 0 else args.hidden_dropout_phase2
            config[f'feature_dropout_phase{phase+1}'] = args.feature_dropout_phase1 if phase == 0 else args.feature_dropout_phase2

        elif hirarchical_net[phase] == 'LipschitzMonotonic':
            config[f'first_ActivateLayer_phase{phase+1}'] = LipschitzMonotonicLayer
            config[f'first_hidden_dim_phase{phase+1}'] = 128           
            config[f'shallow_phase{phase+1}'] = False
            config[f'hidden_ActivateLayer_phase{phase+1}'] = LipschitzMonotonicLayer
            config[f'hidden_dim_phase{phase+1}'] = [128, 64]
            config[f'hidden_dropout_phase{phase+1}'] = args.hidden_dropout_phase1 if phase == 0 else args.hidden_dropout_phase2
            config[f'feature_dropout_phase{phase+1}'] = args.feature_dropout_phase1 if phase == 0 else args.feature_dropout_phase2
            config[f'weight_norms_kind_phase{phase+1}'] = args.weight_norms_kind_phase1 if phase == 0 else args.weight_norms_kind_phase2
            config[f'group_size_phase{phase+1}'] = args.group_size_phase1 if phase == 0 else args.group_size_phase2
            config[f'monotonic_constraint_phase{phase+1}'] = args.monotonic_constraint_phase1 if phase == 0 else args.monotonic_constraint_phase2

    # Common configurations
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['num_exp'] = args.num_exp
    config['in_features'] = args.in_features
    config['latent_dim'] = args.latent_dim
    config['output_dim'] = args.output_dim
    config['seed'] = args.seed

    return config