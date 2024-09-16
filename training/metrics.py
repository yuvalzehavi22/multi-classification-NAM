from matplotlib import pyplot as plt
import sklearn
import torch
import torch.nn.functional as F
import wandb


# def calculate_metric(logits,
#                      truths,
#                      regression=True):
#     """Calculates the evaluation metric."""
#     if regression:
#         # root mean squared error
#         # return torch.sqrt(F.mse_loss(logits, truths, reduction="none")).mean().item()
#         # mean absolute error
#         return "MAE", ((logits.view(-1) - truths.view(-1)).abs().sum() / logits.numel()).item()
#     else:
#         # return sklearn.metrics.roc_auc_score(truths.view(-1).tolist(), torch.sigmoid(logits.view(-1)).tolist())
#         return "accuracy", accuracy(logits, truths)


# def accuracy(logits, truths):
#     return (((truths.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / truths.numel()).item()


# def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
#         """This is for evaluation of one or multi-class classification error rate."""
#         X_test = torch.as_tensor(X_test, device=device)
#         y_test = check_numpy(y_test)
#         self.model.train(False)
#         with torch.no_grad():
#             logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
#             logits = check_numpy(logits)
#             if logits.ndim == 1:
#                 pred = (logits >= 0).astype(int)
#             else:
#                 pred = logits.argmax(axis=-1)
#             error_rate = (y_test != pred).mean()
#         return error_rate

# def evaluate_negative_auc(self, X_test, y_test, device, batch_size=4096):
#     X_test = torch.as_tensor(X_test, device=device)
#     y_test = check_numpy(y_test)
#     self.model.train(False)
#     with torch.no_grad():
#         logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
#         logits = check_numpy(logits)
#         auc = roc_auc_score(y_test, logits)

#     return -auc

# def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
#     X_test = torch.as_tensor(X_test, device=device)
#     y_test = check_numpy(y_test)
#     self.model.train(False)
#     with torch.no_grad():
#         prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
#         prediction = check_numpy(prediction)
#         error_rate = ((y_test - prediction) ** 2).mean()
#     error_rate = float(error_rate)  # To avoid annoying JSON unserializable bug
#     return error_rate

# def evaluate_multiple_mse(self, X_test, y_test, device, batch_size=4096):
#     X_test = torch.as_tensor(X_test, device=device)
#     y_test = check_numpy(y_test)
#     self.model.train(False)
#     with torch.no_grad():
#         prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
#         prediction = check_numpy(prediction)
#         error_rate = ((y_test - prediction) ** 2).mean(axis=0)
#     return error_rate.astype(float).tolist()

# def evaluate_ce_loss(self, X_test, y_test, device, batch_size=512):
#     """Evaluate cross entropy loss for binary or multi-class targets.

#     Args:
#         X_test: input features.
#         y_test (numpy Int array or torch Long tensor): the target classes.

#     Returns:
#         celoss (float): the average cross entropy loss.
#     """
#     X_test = torch.as_tensor(X_test, device=device)
#     y_test = check_numpy(y_test)
#     self.model.train(False)
#     with torch.no_grad():
#         logits = (process_in_chunks(self.model, X_test, batch_size=batch_size))
#         y_test = torch.tensor(y_test, device=device)

#         if logits.ndim == 1:
#             celoss = F.binary_cross_entropy_with_logits(logits, y_test.float()).item()
#         else:
#             celoss = F.cross_entropy(logits, y_test).item()
#     celoss = float(celoss)  # To avoid annoying JSON unserializable bug
#     return celoss