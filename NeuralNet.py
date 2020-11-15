import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_hidden_layers=2, hidden_layer_size=12):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = nn.ModuleList([])
        for layer_i in range(num_hidden_layers):
            if layer_i == 0:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden_layer_size),
                        nn.BatchNorm1d(hidden_layer_size),
                        nn.PReLU()
                    )
                )
            else:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layer_size, hidden_layer_size),
                        nn.BatchNorm1d(hidden_layer_size),
                        nn.PReLU()
                    )
                )
        if num_hidden_layers > 0:
            self.head = nn.Linear(hidden_layer_size, output_size)
        else:
            self.head = nn.Linear(input_size, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.type(torch.float)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.head(x)


# Implementation from source:
# class DQN(torch.nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, hidden_dim=12) -> None:
#         """DQN Network
#         Args:
#             input_dim (int): `state` dimension.
#                 `state` is 2-D tensor of shape (n, input_dim)
#             output_dim (int): Number of actions.
#                 Q_value is 2-D tensor of shape (n, output_dim)
#             hidden_dim (int): Hidden dimension in fc layer
#         """
#         super(DQN, self).__init__()
#
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.PReLU()
#         )
#
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.PReLU()
#         )
#
#         self.final = torch.nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Returns a Q_value
#         Args:
#             x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
#         Returns:
#             torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
#         """
#         x = x.type(torch.float)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.final(x)
#
#         return x