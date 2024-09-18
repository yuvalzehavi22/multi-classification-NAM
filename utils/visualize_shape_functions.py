import matplotlib.pyplot as plt
import torch
import wandb

from data_processing.data_loader import SyntheticDatasetGenerator


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

def get_shape_functions_synthetic_data(model, args, num_test_exp=1000):
    _, y_phase1, shape_functions_phase1 = SyntheticDatasetGenerator.get_synthetic_data_phase1(num_test_exp, args.in_features, is_test=True)
    _, shape_functions_phase2 = SyntheticDatasetGenerator.get_synthetic_data_phase2(y_phase1, is_test=True)

    x_values = torch.linspace(0, 3, num_test_exp).reshape(-1, 1)

    input_dim = args.in_features
    output_dim = args.latent_dim
    visualize_combined_gam(model, x_values, input_dim, output_dim, shape_functions_phase1, vis_lat_features=True)

    if args.hierarch_net:
        input_dim = args.latent_dim
        output_dim = args.output_dim
        visualize_combined_gam(model, x_values, input_dim, output_dim, shape_functions_phase2)
    
    return


def visualize_combined_gam(model, x_values, input_dim, output_dim, shape_functions, vis_lat_features=False):
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on

    x_values = x_values.to(device)
    
    # Initialize global min and max for setting the y-limits
    global_min = float('inf')
    global_max = float('-inf')

    # First pass: Determine the global min and max across all predicted and true shape functions
    for i in range(input_dim):
        with torch.no_grad():
            feature_input = x_values
            for j in range(output_dim):
                # Predicted shape function
                if model.hierarch_net:
                    if vis_lat_features:
                        feature_output = model.NAM_features.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                    else:
                        feature_output = model.NAM_output.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()
                else:
                    feature_output = model.NAM_features.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()

                # Get true shape function
                true_feature_output = shape_functions[f'f_{j}_{i}']

                # Update global min and max
                global_min = min(global_min, feature_output.min(), true_feature_output.min())
                global_max = max(global_max, feature_output.max(), true_feature_output.max())

    # Second pass: Plot the predicted and true shape functions with consistent y-limits
    fig, axes = plt.subplots(input_dim, output_dim, figsize=(15, 30))

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
                    feature_output = model.NAM_features.feature_nns[i](feature_input[:, 0])[:, j].cpu().numpy()

                # Get the true shape function for this feature-output pair
                true_feature_output = shape_functions[f'f_{j}_{i}']

                # Plot the predicted shape function
                ax1 = axes[i, j]
                ax1.scatter(x_values.cpu().numpy(), feature_output, label=f'Predicted Feature {i+1}', color='blue', alpha=0.6)
                
                # Plot the true shape function
                ax1.plot(x_values.cpu().numpy(), true_feature_output, label=f'True Feature {i+1}', color='red', linestyle='--')

                # Set labels and title
                ax1.set_title(f'Feature {i+1} to output {j}')
                ax1.set_xlabel('Input')
                ax1.set_ylabel('Output')

                # Set consistent y-limits across all plots
                ax1.set_ylim([global_min * 1.1, global_max * 1.1])  # Adjust the limits slightly for better visualization

                # Add legend
                ax1.legend()

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