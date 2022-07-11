from argparse import ArgumentParser
from typing import List
import torch

from deep_memory_update.layers import DMULayer, StatefulRNN
from deep_memory_update.models.base_model import (
    BaseRnnModel,
    BaseSeq2SeqModel,
)
from deep_memory_update.models.utils import ListOfListAction


class DMUModel(BaseRnnModel):
    model_name = "DMU"

    def __init__(
        self, recurrent_cells: List[List[int]], increased_bias: float,
        learning_rate_rnn: float, **kwargs
    ):
        self.recurrent_cells = recurrent_cells
        self.increased_bias = increased_bias
        self.learning_rate_rnn = learning_rate_rnn
        super(DMUModel, self).__init__(**kwargs)

    def set_recur_layer(self, input_size: int, stateful: bool) -> int:
        for i, cell in enumerate(self.recurrent_cells):
            self.recur_layers.add_module(
                str(i),
                StatefulRNN(
                    DMULayer(
                        input_size=input_size,
                        cells_sizes=cell,
                        increased_bias=self.increased_bias,
                    ),
                    stateful=stateful,
                ),
            )
            input_size = cell[-1]
        return input_size

    def configure_optimizers(self):
        params = [
            {"params": self.recur_layers.parameters(), "lr": self.learning_rate_rnn,
             "weight_decay": self.weight_decay * self.learning_rate_rnn / self.learning_rate},
        ]
        if self.embedding is not None:
            params.append({"params": self.embedding.parameters(),
                           "lr": self.learning_rate, "weight_decay": self.weight_decay})
        if self.dense_layers is not None:
            params.append({"params": self.dense_layers.parameters(), "lr":
                           self.learning_rate, "weight_decay": self.weight_decay})
        optimizer = self.optimizer(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseRnnModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--recurrent-cells",
            dest="recurrent_cells",
            default=[[4]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--increased-bias",
            dest="increased_bias",
            default=0.0,
            type=float,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--lr-rnn",
            "--learning-rate-rnn",
            dest="learning_rate_rnn",
            default=0.01,
            type=float,
            metavar="LR",
            help="initial learning rate for the RNN block",
        )
        return parser


class DMUSeq2SeqModel(BaseSeq2SeqModel):
    model_name = "DMU"

    def __init__(
        self,
        encoder_cells: List[List[int]],
        decoder_cells: List[List[int]],
        increased_bias: float,
        learning_rate_rnn: float,
        **kwargs
    ):
        self.encoder_cells = encoder_cells
        self.decoder_cells = decoder_cells
        self.increased_bias = increased_bias
        self.learning_rate_rnn = learning_rate_rnn
        super(DMUSeq2SeqModel, self).__init__(**kwargs)

    def create_rnn(self, layers, cells, input_size: int, name_prefix: str) -> int:
        for i, cell in enumerate(cells):
            layers.add_module(
                name_prefix + str(i),
                DMULayer(
                    input_size=input_size,
                    cells_sizes=cell,
                    increased_bias=self.increased_bias,
                ),
            )
            input_size = cell[-1]
        return input_size

    def set_encoder_layer(self, input_size: int) -> int:
        return self.create_rnn(
            layers=self.encoder_layers,
            cells=self.encoder_cells,
            input_size=input_size,
            name_prefix="en_",
        )

    def set_decoder_layer(self, input_size: int, hid_size: int) -> int:
        return self.create_rnn(
            layers=self.decoder_layers,
            cells=self.decoder_cells,
            input_size=input_size,
            name_prefix="de_",
        )

    def configure_optimizers(self):
        wd_rnns = self.weight_decay * self.learning_rate_rnn / self.learning_rate
        params = [
            {"params": self.embedding_enc.parameters(), "lr": self.learning_rate,
             "weight_decay": self.weight_decay},
            {"params": self.embedding_dec.parameters(), "lr": self.learning_rate,
             "weight_decay": self.weight_decay},
            {"params": self.dense_layers.parameters(), "lr": self.learning_rate,
             "weight_decay": self.weight_decay},
            {"params": self.encoder_layers.parameters(), "lr": self.learning_rate_rnn,
             "weight_decay": wd_rnns},
            {"params": self.decoder_layers.parameters(), "lr": self.learning_rate_rnn,
             "weight_decay": wd_rnns},
        ]
        optimizer = self.optimizer(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseSeq2SeqModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--encoder-cells",
            dest="encoder_cells",
            default=[[4]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--decoder-cells",
            dest="decoder_cells",
            default=[[4]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--increased-bias",
            dest="increased_bias",
            default=0.0,
            type=float,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--lr-rnn",
            "--learning-rate-rnn",
            dest="learning_rate_rnn",
            default=0.01,
            type=float,
            metavar="LR",
            help="initial learning rate for the RNN block",
        )
        return parser
