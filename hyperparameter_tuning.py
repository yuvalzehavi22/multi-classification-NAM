import os
import optuna
import subprocess
import json
import torch
from joblib import parallel_backend

def objective(trial):
    # Pass the trial ID to make the file unique
    trial_id = trial.number

    # Define the hyperparameters to tune
    featureNN_arch_phase1 = trial.suggest_categorical('featureNN_arch_phase1', ['single_to_multi_output', 'multi_output'])
    featureNN_arch_phase2 = trial.suggest_categorical('featureNN_arch_phase2', ['single_to_multi_output', 'multi_output', 'parallel_single_output'])

    # ------------------------------------------------------
    # -------------- Phase 1 hidden dimension --------------
    # ------------------------------------------------------
    first_hidden_dim_phase1 = trial.suggest_int('first_hidden_dim_phase1', 32, 512)

    # Suggest how many layers to use, up to 3
    num_hidden_layers_phase1 = trial.suggest_int('num_hidden_layers_phase1', 1, 3)
    
    # Suggest hidden sizes for each layer
    hidden_dim_phase1 = []
    for i in range(num_hidden_layers_phase1):
        hidden_dim_phase1.append(trial.suggest_int(f'hidden_dim_phase1_layer_{i+1}', 16, 128))

    # If fewer than 3 layers, remaining layers are set to None or an empty list
    while len(hidden_dim_phase1) < 3:
        hidden_dim_phase1.append(None)
    
    # Build the command dynamically based on the number of layers and their hidden units
    hidden_dim_str_phase1 = ' '.join([str(d) for d in hidden_dim_phase1 if d is not None])
    
    # ------------------------------------------------------
    # -------------- Phase 2 hidden dimension --------------
    # ------------------------------------------------------
    first_hidden_dim_phase2 = trial.suggest_int('first_hidden_dim_phase2', 32, 512)

    # Suggest how many layers to use, up to 3
    num_hidden_layers_phase2 = trial.suggest_int('num_hidden_layers_phase2', 1, 3)
    
    # Suggest hidden sizes for each layer
    hidden_dim_phase2 = []
    for i in range(num_hidden_layers_phase2):
        hidden_dim_phase2.append(trial.suggest_int(f'hidden_dim_phase2_layer_{i+1}', 16, 128))

    # If fewer than 3 layers, remaining layers are set to None or an empty list
    while len(hidden_dim_phase2) < 3:
        hidden_dim_phase2.append(None)
    
    # Build the command dynamically based on the number of layers and their hidden units
    hidden_dim_str_phase2 = ' '.join([str(d) for d in hidden_dim_phase2 if d is not None])

    # Build the command dynamically based on the hyperparameters
    command = f"python run_model.py --WB_project_name 'Hirarchical_NAMs_hyperparam_optimization' " \
                f"--featureNN_arch_phase1 {featureNN_arch_phase1} --featureNN_arch_phase2 {featureNN_arch_phase2} " \
                f"--first_activate_layer_phase1 'ReLU' --hidden_activate_layer_phase1 'ReLU' " \
                f"--first_hidden_dim_phase1 {first_hidden_dim_phase1} --hidden_dim_phase1 {hidden_dim_str_phase1} " \
                f"--first_activate_layer_phase2 'ReLU' --hidden_activate_layer_phase2 'ReLU' " \
                f"--first_hidden_dim_phase2 {first_hidden_dim_phase2} --hidden_dim_phase2 {hidden_dim_str_phase2} " \
                f"--trial_id {trial_id}"
                

    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    # l1_lambda_phase1 = trial.suggest_float('l1_lambda_phase1', 0.0, 1e-5, log=True)
    # l2_lambda_phase1 = trial.suggest_float('l2_lambda_phase1', 0.0, 1e-3, log=True)
    # l2_lambda_phase2 = trial.suggest_float('l2_lambda_phase2', 0.0, 1e-3, log=True)
    # monotonicity_lambda_phase2 = trial.suggest_float('monotonicity_lambda_phase2', 1e-4, 1e-1, log=True)
    # batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    # weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-4)

    # # Build the command dynamically based on the hyperparameters
    # command = f"python run_model.py --WB_project_name 'Hirarchical_NAMs_hyperparam_optimization' " \
    #             f"--featureNN_arch_phase1 {featureNN_arch_phase1} --featureNN_arch_phase2 {featureNN_arch_phase2} " \
    #             f"--first_activate_layer_phase1 {"ReLU"} --hidden_activate_layer_phase1 {"ReLU"} " \
    #             f"--first_hidden_dim_phase1 {first_hidden_dim_phase1} --first_hidden_dim_phase2 {first_hidden_dim_phase2} " \
    #             f"--first_activate_layer_phase2 {"ReLU"} --hidden_activate_layer_phase2 {"ReLU"} " \
    #             f"--learning_rate {learning_rate} --l1_lambda_phase1 {l1_lambda_phase1} " \
    #             f"--l2_lambda_phase1 {l2_lambda_phase1} --monotonicity_lambda_phase2 {monotonicity_lambda_phase2}" \
    #             f"--l2_lambda_phase2 {l2_lambda_phase2} --batch_size {batch_size} " \
    #             f"--epochs {500} --weight_decay {weight_decay} --trial_id {trial_id} "
        
    # Run the command
    subprocess.run(command, shell=True)
    
    # Load the validation loss from the output file (written by run_model.py)
    with open(f"val_loss_trial_{trial_id}.json", "r") as f:
        val_loss = json.load(f)['val_loss']
    
    os.remove(f"val_loss_trial_{trial_id}.json")
    
    return val_loss  # Optuna minimizes this objective

# Run the optimization
# Parallelize trials across GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = 1

# Parallel execution
with parallel_backend("multiprocessing", n_jobs=num_gpus):
    study = optuna.create_study(direction="minimize") # Minimize validation loss
    study.optimize(objective, n_trials=50)

# Print the best trial based on the minimum loss
best_trial = study.best_trial
print(f"\nBest trial: Trial {best_trial.number}\n, Best hyperparameters: {study.best_params}")

# Save the best model for later use
best_model_path = f"best_model_trial_{best_trial.number}.pth"
torch.save(best_trial, best_model_path)