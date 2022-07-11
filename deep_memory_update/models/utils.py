import argparse
import copy
from argparse import ArgumentParser
from typing import Any, List, Tuple

import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, nn, optim
from torch.nn.utils import rnn


class PerplexityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input: Tensor, target: Tensor, y_len: Tensor) -> Tensor:
        input = input.transpose(1, 2)
        return torch.exp(self.cross_entropy_loss(input, target))


class Perplexity(torchmetrics.Metric):
    label = "perplexity"

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("perplexity", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        input = preds.transpose(1, 2)
        self.perplexity += torch.exp(self.cross_entropy_loss(input, target))
        self.total += 1

    def compute(self):
        return self.perplexity / self.total


class Accuracy(torchmetrics.Accuracy):
    label = "acc"

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        mask = target != 0
        _, y_hat_max = torch.max(preds, dim=target.dim())
        super().update(y_hat_max[mask], target[mask])


class PolyphonicLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input: Tensor, target: Tensor, y_len: Tensor) -> Tensor:
        raw_loss = self.bce_loss(input, target).sum(2)
        mask = torch.ones_like(raw_loss)
        for idx, length in enumerate(y_len):
            mask[idx, length:] = 0
        # sum losses for each time step and then average them all
        return raw_loss[mask.bool()].mean()


class _AlwaysEqualYLenMixin(nn.Module):
    def forward(self, input: Tensor, target: Tensor, y_len: Tensor) -> Tensor:
        if torch.any(y_len != y_len[0]):
            raise RuntimeError("All y_len elements must be equal")
        return super().forward(input, target)


class BCEWithLogitsLoss(_AlwaysEqualYLenMixin, nn.BCEWithLogitsLoss):
    pass


class CrossEntropyLoss(_AlwaysEqualYLenMixin, nn.CrossEntropyLoss):
    pass


class MSELoss(_AlwaysEqualYLenMixin, nn.MSELoss):
    pass


class CrossEntropy3dLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, ignore_index=0, **kwargs):
        super().__init__(*args, ignore_index=ignore_index, **kwargs)

    def forward(self, input: Tensor, target: Tensor, _y_len: Tensor) -> Tensor:
        input = input.transpose(1, 2)
        return super().forward(input, target)


def get_loss(name: str, loss_weight: torch.Tensor = None):
    losses = {
        "MSE": MSELoss,
        "CrossEntropy": CrossEntropyLoss,
        "CrossEntropy3d": CrossEntropy3dLoss,
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "PolyphonicLoss": PolyphonicLoss,
        "PerplexityLoss": PerplexityLoss,
    }
    if loss_weight is not None:
        return losses[name](weight=loss_weight)
    else:
        return losses[name]()


def get_metrics(metrics: List[str]):
    metrics_dict = {
        "Accuracy": Accuracy,
        "Perplexity": Perplexity,
    }
    return [metrics_dict[m]() for m in metrics]


def get_optimizer(name: str):
    optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
    return optimizers[name]


def last_items(packed: rnn.PackedSequence, lengths: torch.Tensor) -> torch.Tensor:
    sum_batch_sizes = torch.cat(
        (torch.zeros(2, dtype=torch.int64), torch.cumsum(packed.batch_sizes, 0))
    )
    sorted_lengths = lengths[packed.sorted_indices]
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.shape[0])
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


class ListOfListAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = 0

    def __call__(self, parser, namespace, values, option_string=None):
        if self.size == 0:
            items = []
        else:
            items = getattr(namespace, self.dest, None)
            items = _copy_items(items)
        items.append(values)
        setattr(namespace, self.dest, items)
        self.size += 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _copy_items(items):
    if items is None:
        return []
    if type(items) is list:
        return items[:]
    return copy.copy(items)


class MinimumSaver:
    def __init__(self):
        self.min = {}
        self.sum = {}
        self.batches = {}

    def log(self, key: str, value: float, batch_size: int):
        if key not in self.sum:
            self.sum[key] = 0
            self.batches[key] = 0

        self.sum[key] += value * batch_size
        self.batches[key] += batch_size

    def calculate(self, key):
        new_min = self.sum[key] / self.batches[key]
        if key not in self.min:
            self.min[key] = new_min
        else:
            self.min[key] = min(new_min, self.min[key])

        self.sum.pop(key, None)
        self.batches.pop(key, None)

    def get_min(self):
        return self.min


def enable_changing_arguments():
    argparse.ArgumentParser.__init__.__defaults__ = replace_at_index(
        argparse.ArgumentParser.__init__.__defaults__, 9, "resolve"
    )


def replace_at_index(tup: Tuple, idx: int, val: Any) -> Tuple:
    lst = list(tup)
    lst[idx] = val
    return tuple(lst)


def turn_off_help_info(parser: argparse.ArgumentParser):
    options = [
        "--logger",
        "--checkpoint_callback",
        "--early_stop_callback",
        "--default_root_dir",
        "--process_position",
        "--tpu_cores",
        "--progress_bar_refresh_rate",
        "--check_val_every_n_epoch",
        "--max_steps",
        "--min_steps",
        "--limit_train_batches",
        "--limit_val_batches",
        "--limit_test_batches",
        "--val_check_interval",
        "--log_save_interval",
        "--row_log_interval",
        "--sync_batchnorm",
        "--weights_summary",
        "--weights_save_path",
        "--num_sanity_val_steps",
        "--truncated_bptt_steps",
        "--resume_from_checkpoint",
        "--reload_dataloaders_every_epoch",
        "--replace_sampler_ddp",
        "--terminate_on_nan",
        "--prepare_data_per_node",
        "--amp_backend",
        "--amp_level",
        "--overfit_pct",
    ]

    for opt in options:
        parser.add_argument(opt, help=argparse.SUPPRESS)


class ThresholdedEarlyStopping(EarlyStopping):
    def __init__(
        self,
        threshold: float = 1e-6,
        monitor: str = "early_stop_on",
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        **kwargs
    ):
        super().__init__(monitor, min_delta, patience, verbose, mode, strict)
        self.threshold = threshold

    def _run_early_stopping_check(self, trainer, _pl_module):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(
                logs
            )  # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        if not self.monitor_op(current, self.threshold):
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience

            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = should_stop

    @staticmethod
    def add_callback_specific_args(parser: ArgumentParser):
        parser.add_argument(
            "--es_threshold",
            dest="threshold",
            default=1e-6,
            type=float,
            metavar="S",
            help="threshold for the early stopping",
        )
        parser.add_argument(
            "--thresholded_early_stopping",
            default=True,
            type=str2bool,
            metavar="BOOL",
            help="enable early stopping",
        )
        parser.add_argument(
            "--es_monitor",
            dest="monitor",
            default="loss/val",
            type=str,
            metavar="MONITOR",
            help="value to be monitored by early stopping",
        )
        parser.add_argument(
            "--es_mode",
            dest="mode",
            default="min",
            type=str,
            metavar="MODE",
            help="mode in which ES works",
        )
        parser.add_argument(
            "--es_patience",
            dest="patience",
            default=5,
            type=int,
            metavar="PATIENCE",
            help="patience for the early stopping in epochs",
        )
        return parser
