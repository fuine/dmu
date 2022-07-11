"""
Script running an experiment on addition sequence data, using LSTM model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import NoisySequenceDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import LSTMModel


class LSTM(LSTMModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = LSTMModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="CrossEntropy",
            optimizer="Adam",
            recurrent_cells=[4],
            all_recurrent_outputs=False,
        )
        return parser


if __name__ == "__main__":
    Experiment(LSTM, NoisySequenceDataModule).run()
