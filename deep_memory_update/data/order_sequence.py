"""
Order sequence generator.
Generates dataset containing samples to task of the prediction of sequence's class.
Sequence consists of some random symbols and X and Y symbols.  Their temporal order
(XXX, XXY, XYX, XYY, YXX, YXY, YYX, or YYY) determines the class.

EXPERIMENT 6b
S. Hochreiter i J. Schmidhuber, “Long short-term memory”, Neural computation,
t. 9, nr. 8, s. 1735–1780, 1997
"""
import math
from argparse import ArgumentParser
from typing import Optional, Tuple

import torch
from torch import Tensor

from deep_memory_update.data.data_module import GeneratorDataModule
from deep_memory_update.data.generator import Generator
from deep_memory_update.data.utils import seq_to_one_hot


class OrderSequenceGenerator(Generator):
    """
    Return order sequence dataset.

    Arguments
        min_length: minimal length of sequence

    Returns
        sequences ( >= min_length, 8) 1
    """

    def __init__(self, min_length: int, **kwargs):
        super().__init__(**kwargs)
        if min_length < 6:
            raise RuntimeError("min_length must be at least 6")
        self.min_length = min_length

    def generate(self) -> Tuple[Tensor, Tensor]:
        max_length = self.min_length + math.ceil(self.min_length / 10)
        length = torch.randint(low=self.min_length, high=max_length + 1, size=()).item()
        rands = torch.randint(4, size=(length,))
        rands[0] = torch.tensor(4)
        rands[-1] = torch.tensor(5)
        first = torch.randint(low=0, high=2, size=())
        second = torch.randint(low=0, high=2, size=())
        third = torch.randint(low=0, high=2, size=())
        y = first * 4 + second * 2 + third
        mark = math.ceil(self.min_length / 10)
        rands[torch.randint(low=mark, high=2 * mark, size=())] = first + 6
        rands[torch.randint(low=3 * mark + 3, high=4 * mark + 3, size=())] = second + 6
        rands[torch.randint(low=6 * mark + 6, high=7 * mark + 6, size=())] = third + 6
        x = seq_to_one_hot(rands, depth=8)
        return x, y


class OrderSequenceDataModule(GeneratorDataModule):
    data_name = "order-sequence"
    pad_sequence = True

    def __init__(self, min_length: int, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    def create_generator(self) -> Generator:
        return OrderSequenceGenerator(min_length=self.min_length)

    def input_size(self) -> int:
        return 8

    def output_size(self) -> Optional[int]:
        return 8

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GeneratorDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--min-length",
            dest="min_length",
            default=100,
            type=int,
            metavar="LEN",
            help="minimal sequence length.",
        )
        return parser
