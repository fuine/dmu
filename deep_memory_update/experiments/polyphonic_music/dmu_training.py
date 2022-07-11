"""
Script running an experiment on addition sequence data, using DMU model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import PolyphonicMusicDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import DMUModel


class DMU(DMUModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = DMUModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="PolyphonicLoss",
            optimizer="Adam",
            recurrent_cells=[36],
            all_recurrent_outputs=True,
        )
        return parser


if __name__ == "__main__":
    Experiment(DMU, PolyphonicMusicDataModule).run()
