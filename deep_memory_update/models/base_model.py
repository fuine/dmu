from abc import ABCMeta
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.nn.utils import rnn
from torch.nn.utils.rnn import PackedSequence

from deep_memory_update.data.utils import SOS_token, PAD_token
from deep_memory_update.models.utils import (
    MinimumSaver,
    get_loss,
    get_metrics,
    get_optimizer,
    str2bool,
)

from torchmetrics.functional import bleu_score
from itertools import islice


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    model_name = ""

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int],
        loss_function: str,
        loss_weight: Tensor,
        learning_rate: float,
        optimizer: str,
        weight_decay: float,
        scheduler_gamma: float,
        alpha_lr: float,
        metrics: List[str],
        **kwargs,
    ):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.loss_function = get_loss(loss_function, loss_weight)
        self.learning_rate = learning_rate
        self.optimizer = get_optimizer(optimizer)
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.metrics = nn.ModuleList(get_metrics(metrics))
        self.alpha = alpha_lr

        self.min_loss = MinimumSaver()

    def forward(self, x: Tensor, y: Tensor, x_len, y_len) -> Tensor:
        raise NotImplementedError

    def step(self, batch, **kwargs):
        x, y, x_len, y_len = batch
        y_hat = self(x, y, x_len, y_len, **kwargs)
        loss = self.loss_function(y_hat, y, y_len)
        for metric in self.metrics:
            metric(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for metric in self.metrics:
            self.log(f"{metric.label}/train_avg", metric, on_step=False, on_epoch=True)
        self.log("loss/train_avg", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        loss = self.step(batch, **kwargs)
        for metric in self.metrics:
            self.log(f"{metric.label}/val", metric, prog_bar=True)
        self.log("loss/val", loss, prog_bar=True)
        self.min_loss.log("loss/val_min", loss.item(), batch[0].shape[0])
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.min_loss.calculate("loss/val_min")
        self.log(
            "loss/val_min",
            self.min_loss.get_min()["loss/val_min"],
            prog_bar=True,
            logger=False,
        )

    def test_step(self, batch, batch_idx, **kwargs):
        loss = self.step(batch, **kwargs)
        for metric in self.metrics:
            self.log(f"{metric.label}/test", metric, prog_bar=True)
        self.log("loss/test", loss)
        return loss

    def on_fit_end(self) -> None:
        if isinstance(self.logger, TensorBoardLogger):
            # TensorBoardLogger does not always flush the logs.
            # To ensure this, we run it manually
            self.logger.experiment.flush()

    def on_test_epoch_end(self):
        for key, value in self.min_loss.get_min().items():
            self.logger.log_hyperparams({key: value})

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--loss-function",
            dest="loss_function",
            default="MSE",
            type=str,
            metavar="LOSS",
            help="name of loss function",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            dest="learning_rate",
            default=0.01,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        parser.add_argument(
            "--optimizer",
            dest="optimizer",
            default="Adam",
            type=str,
            metavar="OPT",
            help="name of optimizer",
        )
        parser.add_argument(
            "--weight-decay",
            dest="weight_decay",
            default=0.0,
            type=float,
            metavar="FLOAT",
            help="weight decay",
        )
        parser.add_argument(
            "--scheduler-gamma",
            dest="scheduler_gamma",
            default=1.0,
            type=float,
            metavar="FLOAT",
            help="scheduler gamma",
        )
        parser.add_argument(
            "--alpha-lr",
            dest="alpha_lr",
            default=1 / 3,
            type=float,
            metavar="FLOAT",
            help="alpha",
        )
        parser.add_argument(
            "--metrics",
            default=[],
            nargs="+",
            type=str,
            metavar="METRIC",
            help="list of metrics to be logged",
        )
        return parser


class BaseRnnModel(BaseModel):
    def __init__(
        self,
        embedding_size: Optional[int],
        stateful: bool,
        all_recurrent_outputs: bool,
        dense_cells: List[int],
        pad_sequence: bool,
        **kwargs,
    ):
        super(BaseRnnModel, self).__init__(**kwargs)
        self.all_recurrent_outputs = all_recurrent_outputs
        self.pad_sequence = pad_sequence

        input_size = self.input_size

        self.embedding = None
        input_size = self.set_embedding_layer(input_size, embedding_size)

        self.recur_layers = nn.ModuleList()
        input_size = self.set_recur_layer(input_size, stateful)

        self.dense_layers = None
        if self.output_size is not None:
            self.set_dense_layer(input_size, dense_cells)

    def set_embedding_layer(self, input_size: int, embedding_size: int) -> int:
        if embedding_size is not None:
            self.embedding = nn.Embedding(
                num_embeddings=input_size, embedding_dim=embedding_size
            )
            input_size = embedding_size
        return input_size

    def set_recur_layer(self, input_size: int, stateful: bool) -> int:
        raise NotImplementedError

    def set_dense_layer(self, input_size: int, dense_cells: List[int]):
        self.dense_layers = nn.Sequential()

        for i, cell in enumerate(dense_cells):
            self.dense_layers.add_module(
                str(i),
                nn.Linear(in_features=input_size, out_features=cell),
            )
            input_size = cell
            self.dense_layers.add_module(
                f"{str(i)}_relu",
                nn.ReLU(),
            )

        self.dense_layers.add_module(
            "out",
            nn.Linear(in_features=input_size, out_features=self.output_size),
        )

    def forward(self, x: Tensor, y: Tensor, x_len, y_len) -> Tensor:
        #  self.old_weights = [p.clone() for p in self.recur_layers.parameters()]
        if self.embedding is not None:
            x = self.embedding(x)

        if self.pad_sequence:
            x = rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=True)

        for recur_layer in self.recur_layers:
            x, h = recur_layer(x)

        if self.all_recurrent_outputs:
            if self.pad_sequence:
                x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        else:
            """last output"""
            x = h[0]

        if self.output_size is not None:
            x = self.dense_layers(x)
        return x

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.reset_state()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.reset_state()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.reset_state()

    def reset_state(self) -> None:
        for module in self.recur_layers.children():
            module.reset_state()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--embedding-size",
            dest="embedding_size",
            default=None,
            type=int,
            metavar="SIZE",
            help="embedding size. If None embedding layer wont be added",
        )
        parser.add_argument(
            "--stateful",
            dest="stateful",
            const=True,
            default=False,
            type=str2bool,
            nargs="?",
            metavar="BOOL",
            help="dense activation function name",
        )
        parser.add_argument(
            "--all-recurrent-outputs",
            dest="all_recurrent_outputs",
            default=False,
            type=bool,
            metavar="BOOLEAN",
            help="use all outputs from last recurrent layer",
        )
        parser.add_argument(
            "--dense-cells",
            dest="dense_cells",
            default=[],
            nargs="+",
            type=int,
            metavar="CELL",
            help="list of additional dense cells",
        )
        return parser


class BaseSeq2SeqModel(BaseModel):
    def __init__(
        self,
        embedding_enc_size: int,
        embedding_dec_size: int,
        teacher_forcing_ratio: float,
        **kwargs,
    ):
        super(BaseSeq2SeqModel, self).__init__(**kwargs)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embedding_enc = nn.Embedding(
            num_embeddings=self.input_size, embedding_dim=embedding_enc_size
        )
        self.encoder_layers = nn.ModuleList()
        enc_hid_size = self.set_encoder_layer(embedding_enc_size)

        self.embedding_dec = nn.Embedding(
            num_embeddings=self.output_size, embedding_dim=embedding_dec_size
        )
        self.decoder_layers = nn.ModuleList()
        dec_hid_size = self.set_decoder_layer(embedding_dec_size, enc_hid_size)

        self.dense_layers = nn.Linear(
            in_features=dec_hid_size, out_features=self.output_size
        )

        self.bleu_ground_truth = []
        self.bleu_outputs = []

    def set_encoder_layer(self, input_size: int) -> int:
        raise NotImplementedError

    def set_decoder_layer(self, input_size: int, hid_size: int) -> int:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor, x_len, y_len, save_outputs=False) -> Tensor:
        self.old_enconder_weights = [
            p.clone() for p in self.encoder_layers.parameters()
        ]
        self.old_decoder_weights = [p.clone() for p in self.decoder_layers.parameters()]

        # Encoder
        x = self.embedding_enc(x)
        x = rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=True)
        for recur_layer in self.encoder_layers:
            x, hx = recur_layer(x)

        # prepare target input
        sos = torch.tensor(SOS_token[1], device=y.device).long().expand(y.shape[0], 1)
        y = torch.cat((sos, y[:, :-1]), dim=1)
        if save_outputs:
            self.previous_gt = [y[i, :j].tolist() for i, j in enumerate(y_len)]
        y = rnn.pack_padded_sequence(
            y, y_len.cpu(), batch_first=True, enforce_sorted=False
        )
        input, batch_sizes, sorted_indices, unsorted_indices = y
        inputs = torch.split(input, batch_sizes.tolist())

        # Decoder
        state = tuple([h[sorted_indices] for h in hx])
        out = None
        outputs = []
        for inp, batch_size in zip(inputs, batch_sizes):
            state = tuple([s[:batch_size] for s in state])
            if out is not None and (
                not self.training or torch.rand(1).item() < self.teacher_forcing_ratio
            ):
                inp = out.argmax(1)[:batch_size]
            inp = self.embedding_dec(inp).unsqueeze(1)

            for recur_layer in self.decoder_layers:
                inp, state = recur_layer(inp, state)

            out = inp.squeeze(1)
            out = self.dense_layers(out)
            outputs += out

        x = PackedSequence(
            torch.stack(outputs), batch_sizes, sorted_indices, unsorted_indices
        )
        x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        if save_outputs:
            outputs = torch.argmax(x, dim=2)
            self.previous_outputs = [
                outputs[i, :j].tolist() for i, j in enumerate(y_len)
            ]

        return x

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx, save_outputs=True)
        self.bleu_outputs.extend(self.previous_outputs)
        self.bleu_ground_truth.extend(self.previous_gt)
        self.previous_outputs = None
        self.previous_gt = None
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        i2w = self.trainer.datamodule.raw_dataset.tar_dic.id2word
        i2w = dict(enumerate(i2w))
        self.bleu_outputs = [
            [i2w.get(x, PAD_token[0]) for x in sentence[1:]]
            for sentence in self.bleu_outputs
        ]
        gts = []
        for sentence in self.bleu_ground_truth:
            sentence_strings = [i2w[x] for x in sentence[1:]]
            assert PAD_token[0] not in sentence_strings
            gts.append([sentence_strings])
        bleu = bleu_score(self.bleu_outputs, gts)
        self.log("bleu/val", bleu, prog_bar=True)
        self.bleu_outputs = []
        self.bleu_ground_truth = []

    def test_step(self, batch, batch_idx):
        loss = super().test_step(batch, batch_idx, save_outputs=True)
        self.bleu_outputs.extend(self.previous_outputs)
        self.bleu_ground_truth.extend(self.previous_gt)
        self.previous_outputs = None
        self.previous_gt = None
        return loss

    def test_epoch_end(self, outputs: List[Any]) -> None:
        i2w = self.trainer.datamodule.raw_dataset.tar_dic.id2word
        i2w = dict(enumerate(i2w))
        self.bleu_outputs = [
            [i2w.get(x, PAD_token[0]) for x in sentence[1:]]
            for sentence in self.bleu_outputs
        ]
        gts = []
        for sentence in self.bleu_ground_truth:
            sentence_strings = [i2w[x] for x in sentence[1:]]
            assert PAD_token[0] not in sentence_strings
            gts.append([sentence_strings])
        bleu = bleu_score(self.bleu_outputs, gts)
        self.log("bleu/test", bleu, prog_bar=True)
        self.bleu_outputs = []
        self.bleu_ground_truth = []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseModel.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--embedding-enc-size",
            dest="embedding_enc_size",
            default=100,
            type=int,
            metavar="SIZE",
            help="embedding size",
        )
        parser.add_argument(
            "--embedding-dec-size",
            dest="embedding_dec_size",
            default=100,
            type=int,
            metavar="SIZE",
            help="embedding size",
        )
        parser.add_argument(
            "--teacher-forcing-ratio",
            dest="teacher_forcing_ratio",
            default=0.5,
            type=float,
            metavar="RATIO",
            help="teacher forcing ratio. value from 0 to 1",
        )
        return parser
