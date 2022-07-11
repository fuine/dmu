"""
Script running an experiment on addition sequence data, using GRU model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import AdditionSequenceDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import DeepGRUModel


class DeepGRU(DeepGRUModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = DeepGRUModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            recurrent_cells=[4],
            all_recurrent_outputs=False,
        )
        return parser


if __name__ == "__main__":
    Experiment(DeepGRU, AdditionSequenceDataModule).run()
