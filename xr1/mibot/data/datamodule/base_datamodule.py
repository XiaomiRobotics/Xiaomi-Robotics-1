# Copyright (C) 2026 Xiaomi Corporation.
import os
from copy import deepcopy

import torch
from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate()
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = JsonDataset(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        generator = torch.Generator()
        generator.manual_seed(
            int(os.environ.get("RANK", 0)) + int(self.params.trainer.get("seed", 42))
        )
        sampler = DistributedSampler(self.train_set, shuffle=True, seed=42)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=8,
            prefetch_factor=4,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=generator,
        )


def _seed_worker(worker_id: int) -> None:
    """Seed each dataloader worker deterministically for reproducibility."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    import random

    random.seed(worker_seed)
    import numpy as np

    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
