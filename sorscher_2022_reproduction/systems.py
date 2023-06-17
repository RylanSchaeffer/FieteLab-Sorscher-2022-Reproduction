import torch
import torch.nn


def create_activation_fn(activation_str: str) -> torch.nn.Module:
    activation_str = activation_str.lower()
    if activation_str == 'shifted_elu':
        class ShiftedELU(torch.nn.Module):

            def __init__(self):
                super(ShiftedELU, self).__init__()
                self.elu = torch.nn.ELU()

            def forward(self, x: torch.Tensor):
                return self.elu(x) + 1.0

        activation_fn = ShiftedELU()
    elif activation_str == 'gelu':
        activation_fn = torch.nn.GELU()
    elif activation_str == 'shifted_gelu':
        class ShiftedGELU(torch.nn.Module):

            def __init__(self):
                super(ShiftedGELU, self).__init__()
                self.gelu = torch.nn.GELU()

            def forward(self, x: torch.Tensor):
                return self.gelu(x) + 0.17

        activation_fn = ShiftedGELU()
    elif activation_str == 'identity':
        class Identity(torch.nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x: torch.Tensor):
                return x

        activation_fn = Identity()

    elif activation_str == 'leaky_relu':
        activation_fn = torch.nn.LeakyReLU()
    elif activation_str == 'relu':
        activation_fn = torch.nn.ReLU()
    elif activation_str == 'sigmoid':
        activation_fn = torch.nn.Sigmoid()
    elif activation_str == 'silu':
        activation_fn = torch.nn.SiLU()
    elif activation_str == 'softmax':
        activation_fn = torch.nn.Softmax(dim=2)
    elif activation_str.startswith('softplus'):
        beta = float(activation_str.split(' ')[1])
        activation_fn = torch.nn.Softplus(beta=beta)
    elif activation_str == 'tanh':
        activation_fn = torch.nn.Tanh()
    else:
        raise NotImplementedError
    return activation_fn

