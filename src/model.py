import torch
import torch.nn as nn

# source https://github.com/MaxRobinsonTheGreat/mandelbrotnn/blob/main/src/models.py
class SkipConn(nn.Module):
    """
    Linear torch model with skip connections between every hidden layer
    and the original input appended to every layer.
    Each hidden layer contains `2*hidden_size+2` params due to skip connections.
    Uses LeakyReLU activations and one final Tanh activation.

    Parameters: 
    - hidden_size (int): Number of non-skip parameters per hidden layer.
    - num_hidden_layers (int): Number of hidden layers.
    """
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, linmap=None):
        super(SkipConn, self).__init__()

        self.inLayer = nn.Linear(init_size, hidden_size)
        self.relu = nn.LeakyReLU()

        hidden_layers = []
        for i in range(num_hidden_layers):
            in_size = (hidden_size * 2 + init_size) if i > 0 else (hidden_size + init_size)
            hidden_layers.append(nn.Linear(in_size, hidden_size))
        self.hidden = nn.ModuleList(hidden_layers)

        self.outLayer = nn.Linear(hidden_size * 2 + init_size, 1)
        self.sig = nn.Sigmoid()
        self._linmap = linmap

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)

        current_output = self.relu(self.inLayer(x))
        previous_output = torch.tensor([]).cuda() if x.is_cuda else torch.tensor([])

        for layer in self.hidden:
            combined = torch.cat([current_output, previous_output, x], dim=1)
            previous_output = current_output
            current_output = self.relu(layer(combined))

        y = self.outLayer(torch.cat([current_output, previous_output, x], dim=1))
        # Using (tanh(y)+1)/2 to map output values between 0 and 1
        return (torch.tanh(y) + 1) / 2


