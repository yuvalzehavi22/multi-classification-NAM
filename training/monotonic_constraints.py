import torch
import torch.nn.functional as F


class MonotonicityEnforcer:
    def __init__(self, input_data, logits, epsilon=1e-3, force_increasing=True):
        self.input = input_data            # [B, D]
        self.logits = logits               # [B, O, D]
        self.epsilon = epsilon
        self.force_increasing = force_increasing

    def compute_penalty(self):
        B, O, D = self.logits.shape
        total_penalty = 0.0

        for d in range(D):
            x_d = self.input[:, d]                         # [B]
            sorted_idx = torch.argsort(x_d)
            logits_d = self.logits[:, :, d]                # [B, O]
            sorted_logits = logits_d[sorted_idx]           # [B, O]
            diffs = sorted_logits[1:] - sorted_logits[:-1]  # [B-1, O]

            if self.force_increasing:
                direction = 1.0
            else:
                direction = torch.where(
                    torch.sum(diffs > 0, dim=0) >= torch.sum(diffs < 0, dim=0),
                    torch.tensor(1.0, device=diffs.device),
                    torch.tensor(-1.0, device=diffs.device)
                )

            penalty = F.relu(-direction * diffs - self.epsilon).sum()
            total_penalty += penalty

        return total_penalty / B


# class MonotonicityEnforcer:
#     def __init__(self, input_data, logits):
#         # Input and logits data
#         self.input = input_data.T
#         self.logits = logits.T
        
#         # Prepare matrices where each input feature is concatenated with its logits
#         self.matrices = self._prepare_matrices()

#     def _prepare_matrices(self):
#         # Concatenate input features with the corresponding logitss
#         return [torch.cat((self.input[feature].unsqueeze(0), self.logits), dim=0) 
#                 for feature in range(self.input.size(0))]
    
#     def sort_matrix_by_first_row(self, matrix):
#         # Sort matrix columns according to the first row (input feature values)
#         sorted_indices = torch.argsort(matrix[0, :])
#         return matrix[:, sorted_indices]

#     def sort_all_matrices(self):
#         # Sort all feature matrices by their first row
#         return [self.sort_matrix_by_first_row(matrix) for matrix in self.matrices]

#     def calculate_monotonicity_penalty(self, sorted_matrices):
#         # Calculate the monotonicity penalty across all features and logitss
#         total_penalty = 0
#         num_examples = sorted_matrices[0].size(1)
        
#         for feature_matrix in sorted_matrices:
#             for logits_idx in range(1, feature_matrix.size(0)):  # Iterate over logits rows
#                 logits_values = feature_matrix[logits_idx, :]
#                 # diffs = logits_values[:-1] - logits_values[1:]  # Differences between consecutive values
#                 # penalty = F.relu(diffs).sum()  # Penalize differences
                
#                 # Differences between consecutive values
#                 diffs = torch.diff(logits_values) # logits_values[1:] - logits_values[:-1]
#                 # Identify monotonicity violations
#                 penalty_mono = 0
#                 for i in range(1,len(diffs)):
#                     penalty_mono += F.relu(-(diffs[i-1]*diffs[i]))

#                 total_penalty += penalty_mono

#         # Normalize penalty by the number of examples
#         total_penalty = total_penalty / num_examples
#         return total_penalty

#     def compute_penalty(self):
#         # Sort matrices and compute the monotonicity penalty
#         sorted_matrices = self.sort_all_matrices()
#         return self.calculate_monotonicity_penalty(sorted_matrices)