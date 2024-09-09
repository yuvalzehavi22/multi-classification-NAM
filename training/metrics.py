import sklearn
import torch
import torch.nn.functional as F


def l2_penalty(params, l2_lambda =0.):
    return l2_lambda * (params ** 2).sum() / params.shape[1]

def l1_penalty(params, l1_lambda):
    l1_norm =  torch.stack([torch.linalg.norm(p, 1) for p in params], dim=0).sum()
    return l1_lambda*l1_norm

def penalized_binary_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    return F.binary_cross_entropy_with_logits(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

def penalized_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    return F.cross_entropy(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

def penalized_mse(logits, truth, fnn_out, feature_penalty=0.):
    return F.mse_loss(logits.view(-1), truth.view(-1)) + l2_penalty(fnn_out, feature_penalty)

# def get_loss_function(task_type):
#     """
#     Selects the appropriate loss function based on the task type.
    
#     Parameters:
#     ----------
#     task_type : str
#         Type of the task: 'regression', 'binary_classification', 'multi_classification'.
        
#     Returns:
#     -------
#     loss_fn : function
#         The selected loss function.
#     """
#     if task_type == 'binary_classification':
#         return penalized_binary_cross_entropy
#     elif task_type == 'multi_classification':
#         return penalized_cross_entropy
#     elif task_type == 'regression':
#         return penalized_mse
#     else:
#         raise ValueError(f"Unknown task type: {task_type}")


def calculate_metric(logits,
                     truths,
                     regression=True):
    """Calculates the evaluation metric."""
    if regression:
        # root mean squared error
        # return torch.sqrt(F.mse_loss(logits, truths, reduction="none")).mean().item()
        # mean absolute error
        return "MAE", ((logits.view(-1) - truths.view(-1)).abs().sum() / logits.numel()).item()
    else:
        # return sklearn.metrics.roc_auc_score(truths.view(-1).tolist(), torch.sigmoid(logits.view(-1)).tolist())
        return "accuracy", accuracy(logits, truths)


def accuracy(logits, truths):
    return (((truths.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / truths.numel()).item()