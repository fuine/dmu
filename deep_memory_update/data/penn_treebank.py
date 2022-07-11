import warnings
from argparse import ArgumentParser
from typing import Optional

import torch
from torch.utils.data import TensorDataset
from torchnlp.datasets import penn_treebank_dataset

from deep_memory_update.data.data_module import BaseDataModule
from deep_memory_update.data.utils import Dictionary

warnings.filterwarnings(
    "ignore",
    message=r"(.*?) class will be retired in the 0.8.0 release and moved to torchtext.legacy",
    category=UserWarning,
)


class PennTreebankDataModule(BaseDataModule):
    data_name = "penn_treebank"
    pad_sequence = False

    def __init__(self, seq_length: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.dictionary = None

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        train, val, test = penn_treebank_dataset(train=True, dev=True, test=True)
        train = process_raw_data(train, self.batch_size, self.seq_length)
        val = process_raw_data(val, self.batch_size_val, self.seq_length)
        test = process_raw_data(test, self.batch_size_test, self.seq_length)

        self.dictionary = Dictionary([train, val, test])

        self.train_dataset = BPTTDataset(
            train, self.batch_size, self.seq_length, self.dictionary
        )
        self.val_dataset = BPTTDataset(
            val, self.batch_size_val, self.seq_length, self.dictionary
        )
        self.test_dataset = BPTTDataset(
            test, self.batch_size_test, self.seq_length, self.dictionary
        )

    def input_size(self) -> int:
        return 10000

    def output_size(self) -> Optional[int]:
        return 10000

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--seq-length",
            dest="seq_length",
            default=100,
            type=int,
            metavar="LEN",
            help="sequence length.",
        )

        return parser


class BPTTDataset(TensorDataset):
    def __init__(
        self, dataset: list, batch_size: int, bptt: int, dictionary: Dictionary
    ) -> None:
        self.dictionary = dictionary
        tensor = torch.tensor([self.dictionary.word2id[d] for d in dataset]).long()

        x = tensor[:-1].view(-1, bptt).contiguous()
        idx = torch.arange(x.shape[0]).view(batch_size, -1).T.reshape(-1)
        x = x[idx]
        # x[batch_size * 0] is followed by x[batch_size * 1]
        # x[batch_size * 1] is followed by x[batch_size * 2], so shuffle has
        # to be turn off.
        y = tensor[1:].view(-1, bptt).contiguous()
        y = y[idx]

        x_len = torch.tensor(bptt).long().expand(x.shape[0])
        y_len = torch.tensor(bptt).long().expand(x.shape[0])

        super().__init__(x, y, x_len, y_len)


def process_raw_data(raw_data, batch_size, bptt):
    _num = (len(raw_data) - 1) // (batch_size * bptt)
    raw_data = raw_data[: (_num * batch_size * bptt + 1)]
    return raw_data
