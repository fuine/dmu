import argparse
import getpass
import random
import time
import warnings
from argparse import ArgumentParser
from typing import Type

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    MLFlowLogger,
    TensorBoardLogger,
)

from deep_memory_update.data import BaseDataModule
from deep_memory_update.models import BaseModel
from deep_memory_update.models.utils import ThresholdedEarlyStopping

warnings.filterwarnings(
    "ignore",
    message="Your `IterableDataset` has `__len__` defined",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The dataloader, (.*?) does not have many workers",
    category=UserWarning,
)


class Experiment:
    def __init__(
        self,
        model: Type[BaseModel],
        data_module: Type[BaseDataModule],
        parser_default: dict = None,
    ):
        self.model = model
        self.data_module = data_module
        self.early_stopping = ThresholdedEarlyStopping
        self.parser_default = parser_default if parser_default is not None else {}

    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()

        if args.seed is not None:
            pl.seed_everything(args.seed)

        if args.fast_dev_run:
            args.batch_size_val = args.batch_size
            args.batch_size_test = args.batch_size

        if args.distributed_backend == "ddp":
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / max(1, args.gpus))
            args.workers = int(args.workers / max(1, args.gpus))

        data_module = self.data_module(**vars(args))
        model = self.model(
            **vars(args),
            input_size=data_module.input_size(),
            output_size=data_module.output_size(),
            loss_weight=data_module.loss_weight(),
            pad_sequence=data_module.pad_sequence,
        )
        early_stopping = self.early_stopping(**vars(args))

        logger = self.create_logger(logger_name=args.logger_name)
        trainer = pl.Trainer.from_argparse_args(args, logger=logger)
        if args.checkpoint_monitor:
            checkpoint_callback = ModelCheckpoint(
                monitor=args.checkpoint_monitor,
                save_top_k=args.checkpoint_top_k,
                mode=args.checkpoint_mode,
            )
            trainer.callbacks.append(checkpoint_callback)

        if args.thresholded_early_stopping:
            trainer.callbacks.append(early_stopping)
        trainer.logger.log_hyperparams(args)

        start = time.time()
        trainer.fit(model, datamodule=data_module)
        end = time.time()

        if not args.no_evaluate:
            if args.checkpoint_monitor:
                trainer.test(ckpt_path=checkpoint_callback.best_model_path)
            else:
                trainer.test(ckpt_path="best")

        print("Elapsed time:", "%.2f" % (end - start))

    def create_logger(self, logger_name: str = "tb") -> LightningLoggerBase:
        if logger_name == "tb":
            return TensorBoardLogger(
                save_dir="tb_logs",
                name=self.data_module.data_name,
            )
        elif logger_name == "mlf":
            return MLFlowLogger(
                experiment_name=self.data_module.data_name,
                tags={
                    "mlflow.runName": self.model.model_name,
                    "mlflow.user": getpass.getuser(),
                },
            )
        else:
            raise RuntimeError(f"Wrong logger name: {logger_name}")

    def create_parser(self):
        parser = ArgumentParser(add_help=True)
        parser = self.add_trainer_parser(parser)
        parser = self.add_experiment_parser(parser)
        parser = self.data_module.add_model_specific_args(parser)
        parser = self.model.add_model_specific_args(parser)
        parser = self.early_stopping.add_callback_specific_args(parser)
        parser.set_defaults(
            progress_bar_refresh_rate=2,
            **self.parser_default,
        )
        parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        return parser

    def add_trainer_parser(self, parser: ArgumentParser):
        parser = pl.Trainer.add_argparse_args(parser)
        parser.set_defaults(
            deterministic=True,
            max_epochs=100,
        )
        return parser

    def add_experiment_parser(self, parser: ArgumentParser):
        parser.add_argument(
            "--no-evaluate",
            dest="no_evaluate",
            action="store_true",
            help="do not evaluate model on validation set",
        )
        parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            default=random.randrange(1 << 32 - 1),
            help="seed for model training.",
        )
        parser.add_argument(
            "--logger-name",
            dest="logger_name",
            type=str,
            choices=["tb", "mlf"],
            default="tb",
            help="Logger name.",
        )
        parser.add_argument(
            "--checkpoint-monitor",
            dest="checkpoint_monitor",
            type=str,
            default="",
            help="Metric used for checkpointing",
        )
        parser.add_argument(
            "--checkpoint-top-k",
            dest="checkpoint_top_k",
            type=int,
            default=1,
            help="Save top k models",
        )
        parser.add_argument(
            "--checkpoint-mode",
            dest="checkpoint_mode",
            type=str,
            choices=["min", "max"],
            default="min",
            help="Mode for the checkpoint monitoring",
        )
        return parser
