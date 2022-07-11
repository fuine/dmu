"""
Script running an experiment on addition sequence data, using DMU model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import TranslationDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import DMUSeq2SeqModel


class DMU(DMUSeq2SeqModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = DMUSeq2SeqModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="CrossEntropy3d",
            optimizer="Adam",
            embedding_enc_size=100,
            embedding_dec_size=100,
            encoder_cells=[[4]],
            decoder_cells=[[4]],
        )
        return parser


if __name__ == "__main__":
    Experiment(DMU, TranslationDataModule).run()
