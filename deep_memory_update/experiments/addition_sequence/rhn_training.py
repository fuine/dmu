"""
Script running an experiment on addition sequence data, using RHN model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import AdditionSequenceDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import RHNModel


class RHN(RHNModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RHNModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            recurrent_cells=[[4, 1]],
            all_recurrent_outputs=False,
        )
        return parser


if __name__ == "__main__":
    Experiment(RHN, AdditionSequenceDataModule).run()
