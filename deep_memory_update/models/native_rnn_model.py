from argparse import ArgumentParser
from typing import List

from deep_memory_update.layers import GRULayer, LSTMLayer, RNNLayer, StatefulRNN
from deep_memory_update.models.base_model import BaseRnnModel, BaseSeq2SeqModel


class BaseNativeRnnModel(BaseRnnModel):
    rnn_model = None

    def __init__(self, recurrent_cells: List[int], **kwargs):
        self.recurrent_cells = recurrent_cells
        super().__init__(**kwargs)

    def set_recur_layer(self, input_size: int, stateful: bool) -> int:
        for i, cell in enumerate(self.recurrent_cells):
            self.recur_layers.add_module(
                str(i),
                StatefulRNN(
                    self.rnn_model(
                        input_size=input_size,
                        hidden_size=cell,
                        batch_first=True,
                    ),
                    stateful=stateful,
                ),
            )
            input_size = cell
        return input_size

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseRnnModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--recurrent-cells",
            dest="recurrent_cells",
            default=[2, 2],
            nargs="+",
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        return parser


class LSTMModel(BaseNativeRnnModel):
    model_name = "LSTM"
    rnn_model = LSTMLayer


class GRUModel(BaseNativeRnnModel):
    model_name = "GRU"
    rnn_model = GRULayer


class RNNModel(BaseNativeRnnModel):
    model_name = "RNN"
    rnn_model = RNNLayer


class BaseNativeSeq2SeqRnnModel(BaseSeq2SeqModel):
    rnn_model = None

    def __init__(self, encoder_cells: List[int], decoder_cells: List[int], **kwargs):
        self.encoder_cells = encoder_cells
        self.decoder_cells = decoder_cells
        super().__init__(**kwargs)

    def create_rnn(self, layers, cells, input_size: int) -> int:
        for i, cell in enumerate(cells):
            layers.add_module(
                str(i),
                self.rnn_model(
                    input_size=input_size,
                    hidden_size=cell,
                    batch_first=True,
                ),
            )
            input_size = cell
        return input_size

    def set_encoder_layer(self, input_size: int) -> int:
        return self.create_rnn(
            layers=self.encoder_layers, cells=self.encoder_cells, input_size=input_size
        )

    def set_decoder_layer(self, input_size: int, hid_size: int) -> int:
        return self.create_rnn(
            layers=self.decoder_layers, cells=self.decoder_cells, input_size=input_size
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseSeq2SeqModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--encoder-cells",
            dest="encoder_cells",
            default=[2, 2],
            nargs="+",
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        parser.add_argument(
            "--decoder-cells",
            dest="decoder_cells",
            default=[2, 2],
            nargs="+",
            type=int,
            metavar="CELL",
            help="list of recurrent cells",
        )
        return parser


class LSTMSeq2SeqModel(BaseNativeSeq2SeqRnnModel):
    model_name = "LSTM"
    rnn_model = LSTMLayer


class GRUSeq2SeqModel(BaseNativeSeq2SeqRnnModel):
    model_name = "GRU"
    rnn_model = GRULayer


class RNNSeq2SeqModel(BaseNativeSeq2SeqRnnModel):
    model_name = "RNN"
    rnn_model = RNNLayer
