from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
from torch import Tensor
from torch.utils import data

from deep_memory_update.data import utils
from deep_memory_update.data.generator import (
    ConstGeneratorDataset,
    Generator,
    GeneratorDataset,
)


class BaseDataModule(pl.LightningDataModule):
    data_name = ""
    pad_sequence = False

    def __init__(
        self,
        batch_size: int,
        batch_size_val: int,
        batch_size_test: int,
        workers: int,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.workers = workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self, **kwargs):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            **kwargs
        )

    def val_dataloader(self, **kwargs):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.workers,
            **kwargs
        )

    def test_dataloader(self, **kwargs):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.workers,
            **kwargs
        )

    def input_size(self) -> int:
        raise NotImplementedError

    def output_size(self) -> Optional[int]:
        raise NotImplementedError

    def loss_weight(self) -> Optional[Tensor]:
        return None

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            dest="batch_size",
            default=32,
            type=int,
            metavar="SIZE",
            help="batch size of training data.",
        )
        parser.add_argument(
            "--batch-size-val",
            dest="batch_size_val",
            default=32,
            type=int,
            metavar="SIZE",
            help="batch size of validation data.",
        )
        parser.add_argument(
            "--batch-size-test",
            dest="batch_size_test",
            default=32,
            type=int,
            metavar="SIZE",
            help="batch size of test data.",
        )
        parser.add_argument(
            "--workers",
            default=4,
            type=int,
            metavar="W",
            help="number of data loading workers",
        )

        return parser


class GeneratorDataModule(BaseDataModule):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        if self.pad_sequence:
            self.collate_fn = utils.collate_fn_pad
        else:
            self.collate_fn = utils.collate_fn

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        self.train_dataset = GeneratorDataset(
            generator=self.create_generator(), max_steps=self.batch_size * self.steps
        )

        self.val_dataset = ConstGeneratorDataset(
            generator=self.create_generator(),
            size=self.batch_size_val,
            collate_fn=self.collate_fn,
        )

        self.test_dataset = ConstGeneratorDataset(
            generator=self.create_generator(),
            size=self.batch_size_test,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return super().train_dataloader(
            collate_fn=self.collate_fn,
        )

    def create_generator(self) -> Generator:
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--steps",
            default=100,
            type=int,
            metavar="S",
            help="number of step during each epoch.",
        )
        parser.set_defaults(
            batch_size_val=1000,
            batch_size_test=10000,
        )

        return parser
