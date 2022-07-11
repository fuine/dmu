"""
Polyphonic music task dataset builder.
Creates dataset containing samples for the polyphonic music autoregression task.
More information about the dataset: http://www-etud.iro.umontreal.ca/~boulanni/icml2012.
"""

from argparse import ArgumentParser
from typing import Optional

import torch
from scipy.io import loadmat

from deep_memory_update.data import BaseDataModule
from deep_memory_update.data.utils import collate_fn_pad


class PolyphonicMusicDataModule(BaseDataModule):
    data_name = "polyphonic_music"
    pad_sequence = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = None

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = get_polyphonic_dataset()

    def train_dataloader(self, **kwargs):
        return super().train_dataloader(
            shuffle=True,
            collate_fn=collate_fn_pad,
        )

    def val_dataloader(self, **kwargs):
        return super().val_dataloader(
            collate_fn=collate_fn_pad,
        )

    def test_dataloader(self, **kwargs):
        return super().test_dataloader(
            collate_fn=collate_fn_pad,
        )

    def input_size(self) -> int:
        return 88

    def output_size(self) -> Optional[int]:
        return 88

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)

        return parser


def _download_dataset():
    return "data/polyphonic-music/Nottingham.mat"


def _load_dataset(path):
    return loadmat(path)


class PolyphonicDataset:
    def __init__(self, dataset) -> None:
        self.x = [torch.from_numpy(sequence[:-1, :]).float() for sequence in dataset]
        self.y = [torch.from_numpy(sequence[1:, :]).float() for sequence in dataset]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_polyphonic_dataset():
    path_to_file = _download_dataset()
    raw_dataset = _load_dataset(path_to_file)
    all_splits = []
    for split in ["train", "valid", "test"]:
        split = list(raw_dataset[f"{split}data"][0])
        all_splits.append(PolyphonicDataset(split))
    return all_splits
