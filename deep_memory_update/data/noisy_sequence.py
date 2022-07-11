"""
Noisy sequence generator.
Generates dataset containing samples to task of the prediction of the next symbol in a
sequence.  There are two possible samples:
[b, a1, a2, a3, ..., b] and [c, a4, a5, a6, ..., c].
Last element in the sequence can be predicted if the first element is remembered by
network.  Data is prepared as x and y for the network, so:
x=[b, a1, a2, a3, ...], y=[b]

EXPERIMENT 2b
S. Hochreiter i J. Schmidhuber, “Long short-term memory”, Neural computation,
t. 9, nr. 8, s. 1735–1780, 1997
"""
from argparse import ArgumentParser
from typing import Optional, Tuple

import torch
from torch import Tensor

from deep_memory_update.data.data_module import GeneratorDataModule
from deep_memory_update.data.generator import Generator
from deep_memory_update.data.utils import seq_to_one_hot


class NoisySequenceGenerator(Generator):
    """
    Returns noisy sequence dataset.

    Arguments
        symbols: nr of different possible symbols in sequence
        seq_len: length of sequence

    Returns
        sequence (seq_len, symbols) 1
    """

    def __init__(self, symbols: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        if symbols < 3:
            raise RuntimeError("symbols must be at least 3")
        self.symbols = symbols
        self.seq_len = seq_len

    def generate(self) -> Tuple[Tensor, Tensor]:
        seq = torch.randint(self.symbols - 2, size=(self.seq_len,))
        first = torch.randint(low=self.symbols - 2, high=self.symbols, size=())
        seq[0] = first
        x = seq_to_one_hot(seq, self.symbols)
        return x, first


class NoisySequenceDataModule(GeneratorDataModule):
    data_name = "noisy-sequence"
    pad_sequence = False

    def __init__(self, symbols: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.symbols = symbols
        self.seq_len = seq_len

    def create_generator(self) -> Generator:
        return NoisySequenceGenerator(symbols=self.symbols, seq_len=self.seq_len)

    def input_size(self) -> int:
        return self.symbols

    def output_size(self) -> Optional[int]:
        return self.symbols

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GeneratorDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--symbols",
            dest="symbols",
            default=100,
            type=int,
            metavar="SYM",
            help="nr of different possible symbols in sequence.",
        )
        parser.add_argument(
            "--seq-length",
            dest="seq_len",
            default=100,
            type=int,
            metavar="LEN",
            help="length of sequence.",
        )
        return parser
