"""
Script running an experiment on addition sequence data, using RHN model.
"""
from argparse import ArgumentParser

from deep_memory_update.data import TranslationDataModule
from deep_memory_update.experiments.experiment import Experiment
from deep_memory_update.models import RHNSeq2SeqModel


class RHN(RHNSeq2SeqModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RHNSeq2SeqModel.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="CrossEntropy3d",
            optimizer="Adam",
            metrics=["Perplexity", "Accuracy"],
            embedding_enc_size=100,
            embedding_dec_size=100,
            encoder_cells=[[4, 1]],
            decoder_cells=[[4, 1]],
        )
        return parser


if __name__ == "__main__":
    Experiment(RHN, TranslationDataModule).run()
