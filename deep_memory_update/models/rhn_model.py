from argparse import ArgumentParser
from typing import List

from deep_memory_update.layers import RHNLayer, StatefulRNN
from deep_memory_update.models.base_model import (
    BaseRnnModel,
    BaseSeq2SeqModel
)
from deep_memory_update.models.utils import ListOfListAction


class RHNModel(BaseRnnModel):
    model_name = "RHN"

    def __init__(self, recurrent_cells: List[List[int]], **kwargs):
        self.recurrent_cells = recurrent_cells
        super().__init__(**kwargs)

    def set_recur_layer(self, input_size: int, stateful: bool) -> int:
        for i, cell in enumerate(self.recurrent_cells):
            self.recur_layers.add_module(
                str(i),
                StatefulRNN(
                    RHNLayer(
                        input_size=input_size,
                        hidden_size=cell[0],
                        recurrence_depth=cell[1],
                    ),
                    stateful=stateful,
                ),
            )
            input_size = cell[0]
        return input_size

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseRnnModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--recurrent-cells",
            dest="recurrent_cells",
            default=[[4, 1]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cell hidden size and depth (currently only one layer supported)",
        )
        return parser


class RHNSeq2SeqModel(BaseSeq2SeqModel):
    model_name = "RHN"

    def __init__(
        self, encoder_cells: List[List[int]], decoder_cells: List[List[int]], **kwargs
    ):
        self.encoder_cells = encoder_cells
        self.decoder_cells = decoder_cells
        super().__init__(**kwargs)

    def create_rnn(self, layers, cells, input_size: int, name_prefix: str) -> int:
        for i, cell in enumerate(cells):
            layers.add_module(
                name_prefix + str(i),
                RHNLayer(
                    input_size=input_size,
                    hidden_size=cell[0],
                    recurrence_depth=cell[1],
                )
            )
            input_size = cell[0]
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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseSeq2SeqModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--encoder-cells",
            dest="encoder_cells",
            default=[[4, 1]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cell hidden size and depth (currently only one layer supported)",
        )
        parser.add_argument(
            "--decoder-cells",
            dest="decoder_cells",
            default=[[4, 1]],
            nargs="+",
            action=ListOfListAction,
            type=int,
            metavar="CELL",
            help="list of recurrent cell hidden size and depth (currently only one layer supported)",
        )
        return parser
