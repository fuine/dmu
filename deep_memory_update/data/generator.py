from typing import Tuple

from torch import Tensor
from torch.utils import data


class Generator:
    def generate(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class GeneratorDataset(data.Dataset):
    def __init__(self, generator: Generator, max_steps: int):
        super().__init__()
        self.generator = generator
        self.max_steps = max_steps

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.generator.generate()

    def __len__(self):
        return self.max_steps


class ConstGeneratorDataset(data.TensorDataset):
    def __init__(self, generator: Generator, size: int, collate_fn):
        self.generator = generator
        self.size = size
        self.collate_fn = collate_fn
        tensors = self.generate_tensors()
        super().__init__(*tensors)

    def generate_tensors(self):
        generator_dataset = GeneratorDataset(self.generator, max_steps=self.size)
        batch = next(
            iter(
                data.DataLoader(
                    generator_dataset, batch_size=self.size, collate_fn=self.collate_fn
                )
            )
        )
        return batch
