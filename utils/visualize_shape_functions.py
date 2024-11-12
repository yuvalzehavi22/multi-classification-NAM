import matplotlib.pyplot as plt
import torch
import wandb

from data_processing.data_loader import SyntheticDatasetGenerator
# from data_processing.data_loader_new import SyntheticDatasetGenerator


def get_shape_functions(model, args, num_test_exp=1000):
    # Generate input values for plotting
    x_values = torch.linspace(0, 3, num_test_exp).reshape(-1, 1)  # 100 points between -1 and 1

    input_dim = args.in_features
    output_dim = args.latent_dim
    visualize_gam(model, x_values, input_dim, output_dim, vis_lat_features = True)

    if args.hierarch_net:
        input_dim = args.latent_dim
        output_dim = args.output_dim
        visualize_gam(model, x_values, input_dim, output_dim, vis_lat_features = False)

    return

def visualize_gam(model, x_values, input_dim, output_dim, vis_lat_features = False):
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on

    x_values = x_values.to(device)
    
    # Plot learned functions
    fig, axes = plt.subplots(input_dim, output_dim, figsize=(15,30))

    feature_output_max = {} 
    feature_output_min = {}

    for j in range(output_dim):
        feature_output_max[f'output_{j}'] = []
        feature_output_min[f'output_{j}'] = []

    for i in range(input_dim):
        with torch.no_grad():
            feature_input = x_values
            for j in range(output_dim):
                if model.hierarch_net:
                    if vis_lat_features:
                        feature_output = model.NAM_features.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                    else:          
                        feature_output = model.NAM_output.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                else:
                    # plot without hirarchical model
                    feature_output = model.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                    
                feature_output_max[f'output_{j}'].append(max(feature_output)) 
                feature_output_min[f'output_{j}'].append(min(feature_output))

    for i in range(input_dim):
        with torch.no_grad(): 
            for j in range(output_dim):
                ax1 = axes[i, j]
                if model.hierarch_net:
                    if vis_lat_features:
                        feature_output = model.NAM_features.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                    else:          
                        feature_output = model.NAM_output.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                else:
                    # plot without hirarchical model
                    feature_output = model.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                    
                ax1.scatter(x_values.cpu().numpy(), feature_output, label=f'Feature {i+1}')
                ax1.set_title(f'Feature {i+1} to output {j}')
                ax1.set_xlabel('Input')
                ax1.set_ylabel('Output')
                ax1.set_ylim([min(feature_output_min[f'output_{j}'])*1.3, max(feature_output_max[f'output_{j}'])*1.3])

    plt.tight_layout()

    if vis_lat_features:
        fig_name = "Shape functions for phase1"
    else:
        fig_name = "Shape functions for phase2"
    plt.savefig(f"{fig_name}.png")

    # Log the plot to W&B
    wandb.log({fig_name: wandb.Image(f"{fig_name}.png")})
    plt.close()
    
    #plt.show()
    return

def get_shape_functions_synthetic_data(model, args, num_test_exp=1000, only_phase2 = False):

    x_values_phase1, y_phase1, shape_functions_phase1, out_weights = SyntheticDatasetGenerator.get_synthetic_data_phase1(num_exp=num_test_exp, raw_features=args.in_features, num_concepts=args.latent_dim, is_test=True, seed=args.seed)

    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on
    x_values_phase1 = x_values_phase1.to(device)

    logits, latent_features, phase1_gams_out, phase2_gams_out= model(x_values_phase1)

    _, shape_functions_phase2 = SyntheticDatasetGenerator.get_synthetic_data_phase2(latent_features, num_classes=args.output_dim, is_test=True)

    x_values_phase2 = torch.linspace(round(float(latent_features.min())), round(float(latent_features.max())), num_test_exp).reshape(-1, 1).to(device)
    x_values_phase2 = x_values_phase2.repeat(1, latent_features.size(1))

    if not only_phase2:
        x_pred = None
        plot_shape_functions(X=x_values_phase1, shape_functions=shape_functions_phase1, phase_gams_out=phase1_gams_out, x_pred=x_pred, num_features=args.in_features, num_outputs=args.latent_dim, vis_lat_features=True)
    if args.hierarch_net or only_phase2:
        plot_shape_functions(X=x_values_phase2, shape_functions=shape_functions_phase2, phase_gams_out=phase2_gams_out, x_pred=latent_features, num_features=args.latent_dim, num_outputs=args.output_dim, vis_lat_features=False)
    return


def plot_shape_functions(X, shape_functions, phase_gams_out, x_pred=None, num_features=10, num_outputs=4, vis_lat_features=False):
    """
    Plots the learned shape functions for each feature.

    Parameters:
    - X (torch.Tensor): Input feature values, used for the x-axis.
    - shape_functions (dict): Dictionary of true shape functions for each feature-output pair.
    - phase_gams_out (torch.Tensor | dict): The learned shape functions.
    - x_pred (torch.Tensor | None): Predicted feature values for scatter plotting.
    - num_features (int): Number of features.
    - num_outputs (int): Number of outputs.
    - vis_lat_features (bool): Whether to visualize latent features.
    """
    # Ensure `X` is on the CPU and converted to NumPy for plotting
    X = X.cpu().numpy()
    if isinstance(phase_gams_out, torch.Tensor):
        phase_gams_out = phase_gams_out.detach().cpu().numpy()
    if x_pred is not None:
        x_pred = x_pred.detach().cpu().numpy()

    # Determine global min and max for consistent y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    
    for key, func in shape_functions.items():
        global_min = min(global_min, func.min())
        global_max = max(global_max, func.max())

    if isinstance(phase_gams_out, dict):
        for key, func in phase_gams_out.items():
            global_min = min(global_min, func.detach().cpu().numpy().min())
            global_max = max(global_max, func.detach().cpu().numpy().max())
    else:
        global_min = min(global_min, phase_gams_out.min())
        global_max = max(global_max, phase_gams_out.max())

    # Apply a 10% padding for better visualization
    y_min = float(global_min - 0.1 * abs(global_min))
    y_max = float(global_max + 0.1 * abs(global_max))

    # Create a figure with a subplot for each feature
    fig, axes = plt.subplots(num_features, num_outputs, figsize=(15, num_features * 3))

    for feature_idx in range(num_features):
        for output_idx in range(num_outputs):
            # Select the subplot for this feature and output
            ax = axes[feature_idx, output_idx] if num_features > 1 else axes[output_idx]

            # Retrieve learned and true shape functions
            if isinstance(phase_gams_out, dict):
                learned_shape_func = phase_gams_out[f'f_{output_idx}_{feature_idx}'].detach().cpu().numpy()
            else:
                learned_shape_func = phase_gams_out[:, output_idx, feature_idx]

            true_shape_func = shape_functions.get(f'f_{output_idx}_{feature_idx}', None).detach().cpu().numpy()

            # Plot the learned shape function for the current feature and output
            if x_pred is not None:
                ax.scatter(x_pred[:, feature_idx], learned_shape_func, color='blue', alpha=0.6, label="Learned")
            else:
                ax.plot(X[:, feature_idx], learned_shape_func, color='blue', alpha=0.6, label="Learned")

            # Plot the true shape function
            ax.plot(X[:, feature_idx], true_shape_func, color='red', linestyle='--', label="True")

            # Set consistent y-limits across all subplots
            ax.set_ylim([y_min, y_max])

            # Set a title and labels
            ax.set_title(f'Feature {feature_idx} to Output {output_idx}')
            ax.set_xlabel("Input Values")
            ax.set_ylabel("Shape Function Output")
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if vis_lat_features:
        fig_name = "Shape functions for phase1"
    else:
        fig_name = "Shape functions for phase2"
    plt.savefig(f"{fig_name}.png")

    # Log the plot to W&B
    if wandb.run is not None:
        wandb.log({fig_name: wandb.Image(f"{fig_name}.png")})
    else:
        plt.show()
        
    plt.close()

    return