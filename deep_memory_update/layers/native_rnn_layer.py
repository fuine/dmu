from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from deep_memory_update.layers.utils import SqueezeStateRNN, TupleStateRNN


class LSTMLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = SqueezeStateRNN(nn.LSTM(*args, **kwargs))
        self._set_forget_gate_bias()

    def _set_forget_gate_bias(self, value: int = 2) -> None:
        with torch.no_grad():
            for name, weights in self.lstm.named_parameters():
                if "bias" in name:
                    # forget gate is between 1/4 and 1/2 of the bias vector
                    quarter_size = len(weights) // 4
                    weights[quarter_size : 2 * quarter_size] = value

    def forward(
        self, x: Tensor, hx: Optional[Tuple[Tensor]] = None
    ) -> Tuple[Tensor, Tuple]:
        return self.lstm(x, hx)


class GRULayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gru = SqueezeStateRNN(TupleStateRNN(nn.GRU(*args, **kwargs)))

    def forward(
        self, x: Tensor, hx: Optional[Tuple[Tensor]] = None
    ) -> Tuple[Tensor, Tuple]:
        return self.gru(x, hx)


class RNNLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rnn = SqueezeStateRNN(TupleStateRNN(nn.RNN(*args, **kwargs)))

    def forward(
        self, x: Tensor, hx: Optional[Tuple[Tensor]] = None
    ) -> Tuple[Tensor, Tuple]:
        return self.rnn(x, hx)
