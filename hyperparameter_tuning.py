import optuna
import subprocess
import json

def objective(trial):
    # Define the hyperparameters to tune
    featureNN_arch_phase1 = trial.suggest_categorical('featureNN_arch_phase1', ['single_to_multi_output', 'multi_output', 'parallel_single_output'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 500, 2000)


    l1_lambda_phase1 = trial.suggest_float('l1_lambda_phase1', 0, 1e-6, log=True)

    # Build the command dynamically based on the hyperparameters
    command = f"python run_model.py --WB_project_name 'GAMs: synt_data_phase1' --featureNN_arch_phase1 {featureNN_arch_phase1} --learning_rate {learning_rate} --epochs {epochs} --l1_lambda_phase1 {l1_lambda_phase1}"
    
    # Run the command
    subprocess.run(command, shell=True)
    
    # Load the validation loss from the output file (written by run_model.py)
    with open("val_loss.json", "r") as f:
        val_loss = json.load(f)['val_loss']
    
    return val_loss  # Optuna minimizes this objective

# Run the optimization
study = optuna.create_study(direction='minimize')  # Minimize validation loss
study.optimize(objective, n_trials=50)  # Run 50 trials

# Get the best hyperparameters
print("Best hyperparameters:", study.best_params)
