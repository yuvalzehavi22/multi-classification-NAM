import torch
import torch.nn as nn
from utils.model_parser import parse_args

args = parse_args()

def get_defult_architecture_phase1(args, block_layers_type='ReLU'):

    if block_layers_type == 'ReLU':
        args.first_activate_layer_phase1 = 'ReLU'
        args.first_hidden_dim_phase1 = 64
        args.shallow_phase1 = False
        args.hidden_activate_layer_phase1 = 'ReLU'
        args.hidden_dim_phase1 = [64, 32]

    elif block_layers_type == 'shallow_ExU':
        args.first_activate_layer_phase1 = 'ExU'
        args.first_hidden_dim_phase1 = 1024
        args.shallow_phase1 = True
    
    elif block_layers_type == 'Monotonic':
        args.first_activate_layer_phase1 = 'LipschitzMonotonic'
        args.first_hidden_dim_phase1 = 128
        args.shallow_phase1 = False
        args.hidden_activate_layer_phase1 = 'LipschitzMonotonic'
        args.hidden_dim_phase1 = [128, 64]
    
    elif block_layers_type == 'ExU_ReLU':
        args.first_activate_layer_phase1 = 'ExU'
        args.first_hidden_dim_phase1 = 64
        args.shallow_phase1 = False
        args.hidden_activate_layer_phase1 = 'ReLU'
        args.hidden_dim_phase1 = [64, 32]


def get_defult_architecture_phase2(args, block_layers_type='Monotonic'):

    if block_layers_type == 'ReLU':
        args.first_activate_layer_phase2 = 'ReLU'
        args.first_hidden_dim_phase2 = 64
        args.shallow_phase2 = False
        args.hidden_activate_layer_phase2 = 'ReLU'
        args.hidden_dim_phase2 = [64, 32]

    elif block_layers_type == 'shallow_ExU':
        args.first_activate_layer_phase2 = 'ExU'
        args.first_hidden_dim_phase2 = 1024
        args.shallow_phase2 = True
    
    elif block_layers_type == 'Monotonic':
        args.first_activate_layer_phase2 = 'LipschitzMonotonic'
        args.first_hidden_dim_phase2 = 128
        args.shallow_phase2 = False
        args.hidden_activate_layer_phase2 = 'LipschitzMonotonic'
        args.hidden_dim_phase2 = [128, 64]
    
    elif block_layers_type == 'ExU_ReLU':
        args.first_activate_layer_phase2 = 'ExU'
        args.first_hidden_dim_phase2 = 64
        args.shallow_phase2 = False
        args.hidden_activate_layer_phase2 = 'ReLU'
        args.hidden_dim_phase2 = [64, 32]
