import torch
import math

class InterpretableLayer(torch.nn.Module):
    __constants__ = ['in_features']
    in_features: int
    out_features: int
    weight: torch

    def __init__(self, in_features: int) -> None:
        super(InterpretableLayer, self).__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features))
        self.softsign = torch.nn.Softsign()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0)

    def forward(self, input: torch) -> torch:
        #  return input*torch.exp(self.weight) + self.bias  # DONE: take exp away an bias and add softsign
        return input * self.weight


class MonotonicLayer(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch

    def pos_tanh(self, x):
        return torch.tanh(x) + 1.

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fn: str = 'exp') -> None:
        super(MonotonicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.fn = fn
        if fn == 'exp':
            self.pos_fn = torch.exp
        elif fn == 'square':
            self.pos_fn = torch.square
        elif fn == 'abs':
            self.pos_fn = torch.abs
        elif fn == 'sigmoid':
            self.pos_fn = torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            self.pos_fn = self.pos_tanh
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n_in = self.in_features
        if self.fn == 'exp':
            mean = math.log(1./n_in)
        else:
            mean = 0
        torch.nn.init.normal_(self.weight, mean=mean)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -1./n_in, 1./n_in)

    def forward(self, input: torch) -> torch:
        ret = torch.matmul(input, torch.transpose(self.pos_fn(self.weight), 0, 1))
        if self.bias is not None:
            ret = ret + self.bias
        return ret
    

# class MonotonicLayer(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super(MonotonicLayer, self).__init__()
        
#         # Initialize weights and biases, apply positive constraint to weights
#         self.weight = torch.nn.Parameter(torch.abs(torch.randn(in_features, out_features)))
#         self.bias = torch.nn.Parameter(torch.zeros(out_features))
        
#     def forward(self, x):
#         # Ensure weights remain non-negative
#         self.weight.data = torch.abs(self.weight)
        
#         # Standard linear transformation with non-negative weights
#         output = x @ self.weight + self.bias
#         return output