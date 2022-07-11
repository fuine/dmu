from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def call_children(func):
    def wrapper(self, *args, **kwargs):
        for module in self.children():
            if hasattr(module, func.__name__):
                getattr(module, func.__name__)(*args, **kwargs)
        func(self, *args, **kwargs)

    return wrapper


class DecoratorModule:
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def __getattr__(self, name: str):
        return getattr(self.rnn, name)

    @property
    def __class__(self):
        return nn.Module


class StatefulRNN(DecoratorModule):
    def __init__(self, rnn, stateful: bool = False):
        super().__init__(rnn)
        self.stateful = stateful
        self.hx = None

    def __call__(self, x):
        hn = self.attach_hx()
        y, hy = self.rnn(x, hn)
        self.hx = self.detach_hx(hy)
        return y, hy

    def attach_hx(self):
        if not self.stateful or self.hx is None:
            return None

        for h in self.hx:
            h.requires_grad = True

        return self.hx

    def detach_hx(self, hy: Tuple[Tensor]):
        if not self.stateful:
            return None

        return tuple([h.detach() for h in hy])

    def reset_state(self):
        self.hx = None


class SqueezeStateRNN(DecoratorModule):
    def __call__(
        self, x: Tensor, hx: Optional[Tuple[Tensor]]
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        if hx is not None:
            hx = tuple([torch.unsqueeze(h, 0) for h in hx])
        y, hy = self.rnn(x, hx)
        hy = tuple([torch.squeeze(h, 0) for h in hy])
        return y, hy


class TupleStateRNN(DecoratorModule):
    def __call__(
        self, x: Tensor, hx: Optional[Tuple[Tensor]]
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        if hx is not None:
            hx = hx[0]
        y, hy = self.rnn(x, hx)
        return y, (hy,)
