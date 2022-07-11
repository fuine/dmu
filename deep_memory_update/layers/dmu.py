import math
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from deep_memory_update.layers.s_rnn_layer import RNNLayer


class BaseDMUFNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        DMUFNNCell constructor.

        Arguments
            :param input_size: input size
            :param hidden_size: output size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.randn(hidden_size, input_size))
        self.bias = Parameter(torch.randn(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            if len(w.shape) >= 2:
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
            else:
                nn.init.zeros_(w)

    def forward(self, inputs: Tensor) -> Tensor:
        """Do forward pass for one timestep.

        Arguments
            :param inputs: input tensor

        :return: output tensor
        """
        y = torch.mm(inputs, self.weight.t()) + self.bias
        return y


class DMUFNNCell(BaseDMUFNNCell):
    """
    DMUFNNCell can appear zero or more times in DMU Layer, from the start up to the
    DMULastFNNCell.
    Performs a simple forward pass.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        """Do forward pass for one timestep.

        Arguments
            :param inputs: input tensor

        :return: output tensor
        """
        return nn.functional.leaky_relu(super().forward(inputs))


class DMULastFNNCell(BaseDMUFNNCell):
    """
    DMULastFNNCell one time in DMU Layer, just before DMUMemoryCell.
    Performs a simple forward pass. Compared to DMUFNNCell, it does not have activation
    function and each odd number in the bias is incremented by a "increased_bias"
    parameter.
    """

    def __init__(self, input_size: int, hidden_size: int, increased_bias: float):
        """
        DMULastFNNCell constructor.

        Arguments
            :param input_size: input size
            :param hidden_size: output size
            :param increased_bias: float by which odd numbers in the bias will be
                                   increased.
        """
        super().__init__(input_size, hidden_size)
        self.increased_bias = increased_bias
        self._increase_bias()

    def reset_parameters(self):
        for w in self.parameters():
            if len(w.shape) >= 2:
                nn.init.xavier_uniform_(w, gain=1)
            else:
                nn.init.zeros_(w)

    def _increase_bias(self):
        """
        Increase each odd number in the bias. [0, 0, 0, 0] -> [2, 2, 0, 0]

        :return: None
        """
        zeros = torch.zeros(int(self.hidden_size / 2))
        to_add = zeros + self.increased_bias
        to_bias = torch.cat((to_add, zeros))
        with torch.no_grad():
            self.bias = nn.Parameter(self.bias + to_bias)


class DMUMemoryCell(nn.Module):
    """
    DMUMemoryCell has to appear one time in DMU Layer, at the end of it.
    Represents memory because computes state.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        DMUMemoryCell constructor.

        Arguments
            :param input_size: input size
            :param hidden_size:  hidden size. (input_size == 2 * hidden_size)
        """
        super(DMUMemoryCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if self.input_size != self.hidden_size * 2:
            raise ValueError(
                "Last cell ({}) should have exactly 2 times less units than previous cell. "
                "Current sizes: {}, {}".format(
                    self.__class__.__name__, self.input_size, self.hidden_size
                )
            )

    def forward(self, inputs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Do forward pass for one timestep."""
        z, h_hat = inputs.chunk(2, dim=1)
        sigma = torch.sigmoid(z)
        y = state * sigma + torch.tanh(h_hat) * (1 - sigma)
        return y, y

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            batch_size, self.hidden_size, requires_grad=self.training, device=device
        )


class DMULayer(RNNLayer):
    """Deep Memory Updater layer class."""

    def __init__(
        self,
        input_size: int,
        cells_sizes: List[int],
        increased_bias: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        input_size = input_size + cells_sizes[-1]  # input size + memory hidden size

        self.fnn_cells = nn.Sequential()
        for i, cell in enumerate(cells_sizes[:-1]):
            self.fnn_cells.add_module(
                "fnn_cell_{}".format(i),
                DMUFNNCell(input_size=input_size, hidden_size=cell),
            )
            input_size = cell

        self.fnn_cells.add_module(
            "last_fnn_cell",
            DMULastFNNCell(
                input_size=input_size,
                hidden_size=cells_sizes[-1] * 2,
                increased_bias=increased_bias,
            ),
        )
        input_size = cells_sizes[-1] * 2

        self.memory_cell = DMUMemoryCell(
            input_size=input_size, hidden_size=cells_sizes[-1]
        )

    def _init_state(self, batch_size, device):
        return (self.memory_cell.init_state(batch_size, device),)

    def step(self, inp: Tensor, state):
        (hx,) = state
        inp = torch.cat((inp, hx), dim=1)
        inp = self.fnn_cells(inp)
        out, hx = self.memory_cell(inp, hx)
        return out, (hx,)
