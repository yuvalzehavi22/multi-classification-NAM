#!/bin/bash

# --------------------------------------------------
# ---------------- synt_data_phase1 ----------------
# --------------------------------------------------

# comparing only featureNN architecture for phase1
# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'single_to_multi_output' --hierarch_net 0 
# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'multi_output' --hierarch_net 0
# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'parallel_single_output' --hierarch_net 0

# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'single_to_multi_output' --hierarch_net 0 --l2_lambda_phase1 1e-6
#python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'single_to_multi_output' --hierarch_net 0 --lr_scheduler 'StepLR'
#python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'single_to_multi_output' --hierarch_net 0 --l1_lambda_phase1 1e-8

#python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase1 'single_to_multi_output' --hierarch_net 0 --lr_scheduler 'StepLR' --l2_lambda_phase1 1e-6 --l1_lambda_phase1 1e-8


# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" 

# python run_model.py --WB_project_name "GAMs-synt_data_phase1" --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 128 --hidden_dim_phase1 128 64 --first_activate_layer_phase1 "LipschitzMonotonic" --hidden_activate_layer_phase1 "LipschitzMonotonic"

# python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-5 --l1_lambda_phase2 1e-6

# python run_model.py --seed 42 --eval_every 50 --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6




# --------------------------------------------------
# ---------------- synt_data_phase2 ----------------
# --------------------------------------------------

# python run_model.py --WB_project_name "GAMs-synt_data_phase2"



# --------------------------------------------------
# ------------ Hirarchial-GAMs synt_data -----------
# --------------------------------------------------

python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 128 --hidden_dim_phase2 128 64 --first_activate_layer_phase2 "LipschitzMonotonic" --hidden_activate_layer_phase2 "LipschitzMonotonic" --track_gradients 0 --monotonic_constraint_phase2 1
python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 64 --hidden_dim_phase2 64 32 --first_activate_layer_phase2 "ReLU" --hidden_activate_layer_phase2 "ReLU" --monotonicity_lambda_phase2 0.001 --l2_lambda_phase1 0.00001 --l2_lambda_phase2 0.00001

python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 32 --hidden_dim_phase2 32 16 --first_activate_layer_phase2 "ReLU" --hidden_activate_layer_phase2 "ReLU"
python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --first_hidden_dim_phase1 64 --hidden_dim_phase1 64 32 --first_activate_layer_phase1 "ReLU" --hidden_activate_layer_phase1 "ReLU" --first_hidden_dim_phase2 64 --hidden_dim_phase2 64 32 --first_activate_layer_phase2 "ReLU" --hidden_activate_layer_phase2 "ReLU" --l1_lambda_phase1 1e-9 --monotonicity_lambda_phase2 0.001 --lr_scheduler 'StepLR' --StepLR_gamma 0.01 --StepLR_step_size 20
# python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0005 --epochs 1000 --l1_lambda_phase1 1e-8 --monotonicity_lambda_phase2 0.001

# python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-5 --l1_lambda_phase2 1e-6

# python run_model.py --WB_project_name "Hirarchial_GAMs-synt_data" --featureNN_arch_phase1 'single_to_multi_output' --featureNN_arch_phase2 'parallel_single_output' --learning_rate 0.0001 --epochs 1000 --l1_lambda_phase1 1e-8 --l1_lambda_phase2 1e-7 --monotonicity_lambda 1e-6
