import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_hidden_layers=2, hidden_layer_size=64):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = []
        for layer_i in range(num_hidden_layers):
            if layer_i == 0:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden_layer_size),
                        # nn.BatchNorm1d(hidden_layer_size),
                        nn.PReLU()
                    )
                )
            else:
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layer_size, hidden_layer_size),
                        # nn.BatchNorm1d(hidden_layer_size),
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
