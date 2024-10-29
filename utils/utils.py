from matplotlib import pyplot as plt
import torch
import os
import random
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import wandb

def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name

# Ensure deterministic behavior
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_data_histograms(values, values_name, nbins=50, model_predict = False, save_path="data_processing/plots/"):
    """
    Plots histograms for each feature in the dataset and saves them for later exploration (raw features, concepts and targets).

    Parameters:
    -----------
    values : torch.Tensor
        The input tensor containing feature values.
    
    values_name : str
        The input type for the function. options: Input, Concept, Target

    nbins : int, optional (default=50)
        Number of bins for the histograms.

    model_predict : bool
        If True - load the model and get the results of the trained model
    
    save_path : str, optional (default='plots/')
        The path where the plots will be saved.
    """

    # Convert values to a pandas DataFrame for easier handling
    num_features = values.shape[1]
    df = pd.DataFrame(values.detach().cpu().numpy(), columns=[f'feature_{i}' for i in range(num_features)])

    fig = go.Figure()

    for i in range(num_features):
        fig.add_trace(
            go.Histogram(x=df[f'feature_{i}'], name=f'{values_name} {i}', opacity=0.75, nbinsx=nbins)
        )

    if model_predict:
        fig_title = f"predicted {values_name}s histogram"
    else:
        fig_title = f"{values_name}s histogram"

    # Update layout
    fig.update_layout(
        title=fig_title,
        xaxis_title='Value',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay histograms
        bargap=0.2,  # Gap between bars
        showlegend=True
    )

    # Save plot
    save_plot = False
    if save_plot:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plot_filename = os.path.join(save_path, f"{fig_title}.png")
        fig.write_image(plot_filename)

        print(f"Plot saved to {plot_filename}")

    if wandb.run is not None:
        wandb.log({f"data/{fig_title}": fig})

    return fig


def plot_pred_data_histograms(model, hierarch_net, inputs):
    """
    Plots histograms of the predicted output.
    using the "plot_data_histograms" function in the "utils/utils.py" file
    """
    device = define_device("auto")

    model.eval()
    model.to(device)
    
    inputs = inputs.to(device)

    if hierarch_net:
        logits, latent_features, _, _ = model(inputs)
        _ = plot_data_histograms(values=latent_features, values_name='Concept',nbins=80, model_predict=True, save_path="data_processing/plots/")
        _ = plot_data_histograms(values=logits, values_name='Target',nbins=100, model_predict=True, save_path="data_processing/plots/")
    else:
        logits, _ = model(inputs)
        _ = plot_data_histograms(values=logits, values_name='Concept',nbins=80, model_predict=True, save_path="data_processing/plots/")


def plot_concepts_weights(weights_dict, model, model_predict = False):
    """
    Only is phase1 arch is single_to_multi_output!!
    Plots the multi_output_layer weights for each feature in the model.
    
    Parameters:
    - weights_dict: dict
        A dictionary containing the true weights for each raw feature and concept class.
    - model : PyTorch model
        The trained model with multiple NAM blocks, each having its own multi_output_layer.
    - model_predict : bool
        If True, indicates that the weights value is after the model was trained.
    """
    concepts = sorted(set(int(key.split('_')[1]) for key in weights_dict.keys()))
    features = sorted(set(int(key.split('_')[2]) for key in weights_dict.keys()))

    num_concepts = len(concepts)
    num_features = len(features)
    
    fig, axes = plt.subplots(num_concepts, 1, figsize=(10, 4 * num_concepts))

    # If there's only one feature, axes will not be a list; convert it to a list for consistent indexing
    if num_concepts == 1:
        axes = [axes]

    # Iterate through each feature block and plot the multi_output_layer weights
    for i, concept in enumerate(concepts):
        # Extract the true weights for the current class
        true_weights = [weights_dict[f'f_{concept}_{j}'] for j in features]
        
        # Extract the predicted weights from the model for the current class
        predicted_weights = []
        for j in features:
            feature_nn = model.NAM_features.feature_nns[j]
            predicted_weight = feature_nn.multi_output_layer.weight[concept].detach().cpu().item()
            predicted_weights.append(predicted_weight)

        # Define the width of each bar and the positions
        bar_width = 0.35
        x = np.arange(num_features)

        # Create the bar plots for true and predicted weights
        axes[i].bar(x - bar_width / 2, true_weights, bar_width, label='True Weights', color='blue')
        axes[i].bar(x + bar_width / 2, predicted_weights, bar_width, label='Predicted Weights', color='orange')

        # Set the title and labels
        axes[i].set_title(f'Weights for concept {concept}')
        axes[i].set_xlabel('Feature Index')
        axes[i].set_ylabel('Weight Value')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(features)
        axes[i].legend()

    plt.tight_layout()

    # Set the figure title
    fig_title = "True vs. Predicted multi_output_layer Weights" if model_predict else "True vs. Initialize multi_output_layer Weights"
    fig.suptitle(fig_title, fontsize=16)

    # Log the plot to W&B or show the plot if not logging
    if wandb.run is not None:
        wandb.log({f"Weights/{fig_title}": wandb.Image(fig)})
    else:
        plt.show()
    
    plt.close(fig)


#    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))

#     # If there's only one feature, axes will not be a list; convert it to a list for consistent indexing
#     if num_classes == 1:
#         axes = [axes]

#     # Iterate through each feature block and plot the multi_output_layer weights
#     for i, feature_nn in enumerate(model.NAM_features.feature_nns):
#         # Get the multi_output_layer weights for the current feature
#         weights = feature_nn.multi_output_layer.weight.detach().cpu().numpy()  # Shape: [num_classes]
        
#         # Bar plot for the current feature's weights
#         axes[i].bar(range(num_classes), weights.flatten())
#         axes[i].set_title(f'Feature {i} Weights')
#         axes[i].set_xlabel('Classes')
#         axes[i].set_ylabel('Weight value')
#         axes[i].set_xticks(range(num_classes))
    
#     plt.tight_layout()

#     # Set the figure title
#     fig_title = "Predicted multi_output_layer Weights" if model_predict else "multi_output_layer Weights"
#     fig.suptitle(fig_title, fontsize=16)

#     # Log the plot to W&B or show the plot if not logging
#     if wandb.run is not None:
#         wandb.log({f"Weights/{fig_title}": wandb.Image(fig)})
#     else:
#         plt.show()
    
#     plt.close(fig)