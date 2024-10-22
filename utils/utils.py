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