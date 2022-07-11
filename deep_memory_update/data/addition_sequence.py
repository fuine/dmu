"""
Addition sequence generator.
Generates dataset containing samples to task of adding marked symbols in a sequence.
x contains two columns: real number and marks, being -1, 0 or 1.
y contains one column: real number.

EXPERIMENT 4
S. Hochreiter i J. Schmidhuber, “Long short-term memory”, Neural computation,
t. 9, nr. 8, s. 1735–1780, 1997
"""
from __future__ import annotations

import math
from argparse import ArgumentParser
from typing import Optional, Tuple

import torch
from torch import Tensor

from deep_memory_update.data.data_module import GeneratorDataModule
from deep_memory_update.data.generator import Generator


class AdditionSequenceGenerator(Generator):
    """
    Generate addition sequence dataset.
    X has two columns, real numbers and marks. Two samples are marked by 1, rest by 0,
    except for first and last, which are marked by -1. Real numbers marked by 1 are
    summed up and y contains those sums.

    Arguments
        min_length: minimal length of sequence

    Returns
        sequences ( >= min_length, 2) (1)
    """

    def __init__(self, min_length: int, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    def generate(self) -> Tuple[Tensor, Tensor]:
        max_length = self.min_length + math.ceil(self.min_length / 10)
        length = torch.randint(low=self.min_length, high=max_length + 1, size=()).item()
        rands = 2 * torch.rand(length, 1) - 1  # [-1,1]
        marks = torch.zeros(length, 1)
        first = torch.randint(low=0, high=min(10, length - 1), size=())
        second = first
        while second == first:
            second = torch.randint(low=0, high=int(length / 2 - 1), size=())
        marks[first, 0] = 1.0
        marks[second, 0] = 1.0
        marks[0, 0] = -1.0
        marks[-1, 0] = -1.0
        x1 = rands[first, 0] if first.item() > 0 else 0
        x2 = rands[second, 0] if second.item() > 0 else 0
        y = (0.5 + (x1 + x2) / 4).reshape(1)
        x = torch.cat((rands, marks), dim=1)
        return x, y


class AdditionSequenceDataModule(GeneratorDataModule):
    data_name = "addition-sequence"
    pad_sequence = True

    def __init__(self, min_length: int, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    def create_generator(self) -> Generator:
        return AdditionSequenceGenerator(min_length=self.min_length)

    def input_size(self) -> int:
        return 2

    def output_size(self) -> Optional[int]:
        return 1

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
