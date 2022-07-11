from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence


class RNNLayer(nn.Module):
    """Custom RNN layer"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        batch: Union[Tensor, PackedSequence],
        hidden: Optional[Tuple[Tensor, ...]] = None,
    ) -> Tuple[Union[Tensor, PackedSequence], Tuple[Tensor, ...]]:
        """
        Arguments
            :param batch: tensor of shape (batch, seq_len, input_size) or PackedSequence
            :param hidden: hidden state, can be None

        :return: (outputs, state)
            outputs - stacked outputs from all timesteps of rnn network
            hy - last state (output) of the network
        """
        if isinstance(batch, PackedSequence):
            return self.forward_packed_sequence(batch, hidden)
        else:
            return self.forward_tensor(batch, hidden)

    def forward_tensor(
        self, batch: Tensor, hidden: Optional[Tuple[Tensor, ...]]
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Arguments
            :param batch: tensor of shape (batch, seq_len, input_size)
            :param hidden: hidden state, can be None

        :return: (outputs, state)
            outputs - stacked outputs from all timesteps of rnn network
            state - last state (output) of the network
        """
        # bach = [batch size, input len, input size]

        inputs = batch.unbind(1)
        # inputs[i] = [batch size, input size],
        # len(inputs) == input len

        state = self.init_state(batch.shape[0], hidden, batch.device)
        outputs = []
        for inp in inputs:
            out, state = self.step(inp, state)
            # out = [batch size, hid dim]
            # state = tuple( [batch size, hid dim] )
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        # outputs = [batch size, input len, hid dim]
        return outputs, state

    def forward_packed_sequence(
        self, batch: PackedSequence, hidden: Optional[Tuple[Tensor, ...]]
    ) -> Tuple[PackedSequence, Tuple[Tensor, ...]]:
        """
        Arguments
            :param batch: PackedSequence
            :param hidden: hidden state, can be None

        :return: (outputs, state)
            outputs - PackedSequence of outputs from all timesteps of rnn network
            state - last state (output) of the network
        """
        input, batch_sizes, sorted_indices, unsorted_indices = batch
        # input = [sum(batch_sizes), input size]
        # len(batch_sizes) == input len

        inputs = torch.split(input, batch_sizes.tolist())
        # inputs[i] = [batch_sizes[i], input size]
        # len(inputs) == input len

        state = self.init_state(batch_sizes[0], hidden, input.device)
        outputs = []
        for inp, batch_size in zip(inputs, batch_sizes):
            new_state = tuple([torch.narrow(s, 0, 0, batch_size) for s in state])
            out, new_state = self.step(inp, new_state)
            # out = [batch size, hid dim]
            # new_state = tuple( [batch size, hid dim] )
            outputs += out

            if state[0].shape[0] != batch_size:
                state = tuple(
                    [torch.cat((ns, s[batch_size:])) for ns, s in zip(new_state, state)]
                )
            else:
                state = new_state

        return (
            PackedSequence(
                torch.stack(outputs), batch_sizes, sorted_indices, unsorted_indices
            ),
            state,
        )

    def init_state(
        self, batch_size: int, hidden: Optional[Tuple[Tensor, ...]], device
    ) -> Tuple[Tensor]:
        if hidden is None:
            return self._init_state(batch_size, device)

        return hidden

    def _init_state(self, batch_size: int, device) -> Tuple[Tensor]:
        raise NotImplementedError

    def step(
        self, inp: Tensor, state: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError
