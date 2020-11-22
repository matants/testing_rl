import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_hidden_layers=2, hidden_layer_size=16):
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


class DQN_conv(nn.Module):
    def __init__(self, in_channels=3, n_actions=15, height=64, width=64):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc4 = nn.Linear(linear_input_size, 64)
        self.fc5 = nn.Linear(64, 64)
        self.head = nn.Linear(64, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.relu(self.fc5(x))
        return self.head(x)
