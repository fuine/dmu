from typing import Tuple

import torch
from torch import Tensor, nn

from deep_memory_update.layers.s_rnn_layer import RNNLayer


class HighwayFirstLayer(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.linear_h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_t_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_h_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_t_x = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input):
        s, x = input
        h = torch.tanh(self.linear_h_x(x) + self.linear_h_h(s))
        t = torch.sigmoid(self.linear_t_x(x) + self.linear_t_h(s))
        return h * t + s * (1 - t)


class HighwayNextLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_t_h = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, s):
        h = torch.tanh(self.linear_h_h(s))
        t = torch.sigmoid(self.linear_t_h(s))
        return h * t + s * (1 - t)


class RHNCell(nn.Module):
    def __init__(self, input_size, hidden_size, recurrence_depth):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrence_depth = recurrence_depth
        self.layers = nn.Sequential()
        self.layers.add_module("highway_first_layer",
                               HighwayFirstLayer(hidden_size, input_size))
        for i in range(1, recurrence_depth):
            self.layers.add_module("highway_next_layer_" + str(i),
                                   HighwayNextLayer(hidden_size))

    def forward(
        self, input: Tensor, state: Tuple[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        s_i, = state
        s_i = self.layers.forward((s_i, input))
        return s_i, (s_i, )

    def init_state(self, batch_size: int, device: torch.device):
        zero = torch.zeros(
            batch_size, self.hidden_size, requires_grad=self.training, device=device
        )
        return (zero,)


class RHNLayer(RNNLayer):
    def __init__(self, input_size: int, hidden_size: int, recurrence_depth: int):
        super().__init__()

        self.rnn = RHNCell(input_size, hidden_size, recurrence_depth)

    def _init_state(self, batch_size, device):
        return self.rnn.init_state(batch_size, device)

    def step(self, inp: Tensor, state):
        return self.rnn(inp, state)
