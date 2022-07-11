"""
Translation task dataset builder.
Creates dataset containing samples for the translation task (from Spanish to English).
Code comes from https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""

import io
import re
import unicodedata
from argparse import ArgumentParser

import torch

from deep_memory_update.data import BaseDataModule
from deep_memory_update.data.utils import (
    Dictionary,
    EOS_token,
    SOS_token,
    collate_fn_pad,
)


class TranslationDataModule(BaseDataModule):
    data_name = "translation"
    pad_sequence = True

    def __init__(self, lang_pair: str, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = None
        self.lang_pair = lang_pair

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        lang = self.lang_pair.split("-")[0]
        data_path = f"data/translation/{self.lang_pair}/{lang}.txt"
        self.raw_dataset = get_translation_dataset(data_path)

        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(
            self.raw_dataset, train_split=70, val_split=15
        )

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
        if self.lang_pair == "spa-eng":
            return 24793
        elif self.lang_pair == "por-eng":
            return 20655
        elif self.lang_pair == "deu-eng":
            return 34628
        else:
            raise RuntimeError("Unknown lang pair")

    def output_size(self) -> int:
        if self.lang_pair == "spa-eng":
            return 12933
        elif self.lang_pair == "por-eng":
            return 12267
        elif self.lang_pair == "deu-eng":
            return 15781
        else:
            raise RuntimeError("Unknown lang pair")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--lang-pair",
            default="spa-eng",
            type=str,
            metavar="LANG",
            help="pair of languages",
        )
        return parser


def _unicode_to_ascii(s):
    # Converts the unicode file to ascii
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _preprocess_sentence(w):
    w = _unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿¡])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿¡]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = "{} {} {}".format(SOS_token[0], w, EOS_token[0])
    return w.split()


def _load_dataset(path):
    # en_sentence = u"May I borrow this book?"
    # sp_sentence = u"¿Puedo tomar prestado este libro?"
    # print(preprocess_sentence(en_sentence))
    # print(preprocess_sentence(sp_sentence).encode('utf-8'))

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    with io.open(path, encoding="UTF-8") as fp:
        lines = fp.read().strip().split("\n")

    word_pairs = [
        [_preprocess_sentence(w) for idx, w in enumerate(line.split("\t")) if idx < 2]
        for line in lines[:30000]
    ]

    return zip(*word_pairs)


class TranslationDataset:
    def __init__(self, inp_dataset, tar_dataset) -> None:
        self.inp_dic = Dictionary(inp_dataset)
        self.tar_dic = Dictionary(tar_dataset)

        self.x = [
            torch.tensor([self.inp_dic.word2id[w] for w in line])
            for line in inp_dataset
        ]
        self.y = [
            torch.tensor([self.tar_dic.word2id[w] for w in line][1:])
            for line in tar_dataset
        ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_translation_dataset(data_path):
    """
    Return translation task dataset.
    """
    tar_lang, inp_lang = _load_dataset(data_path)

    dataset = TranslationDataset(inp_lang, tar_lang)

    return dataset


def split_dataset(dataset, train_split: int, val_split: int):
    train_len = len(dataset) // 100 * train_split
    val_len = len(dataset) // 100 * val_split
    test_len = len(dataset) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    return train_ds, val_ds, test_ds
