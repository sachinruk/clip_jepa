import dataclasses
import multiprocessing as mp
import platform
from typing import Callable

import datasets
from loguru import logger
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src import config

DataLoaderType = DataLoader[dict[str, list[str] | torch.Tensor]]


@dataclasses.dataclass
class Batch:
    images: torch.Tensor
    texts: list[str]

    def __len__(self) -> int:
        return len(self.images)


def collate_fn(batch: list[dict[str, Image.Image | str]]) -> dict[str, list[Image.Image | str]]:
    return {
        "images": [data_instance["image"] for data_instance in batch],
        "texts": [data_instance["caption"] for data_instance in batch],
    }


class CollateFn:
    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, batch: list[dict[str, Image.Image | str]]) -> Batch:
        stacked_images = torch.stack([self.transform(item["image"]) for item in batch])
        texts: list[str] = [item["caption"] for item in batch]

        return Batch(images=stacked_images, texts=texts)


def _get_dataloaders(
    train_ds: datasets.Dataset,
    valid_ds: datasets.Dataset,
    hyper_parameters: config.HyperParameters,
    collate_fn: Callable[[list[dict[str, Image.Image | str]]], dict[str, list[str] | torch.Tensor]],
) -> tuple[DataLoaderType, DataLoaderType]:
    logger.info("Creating Dataloaders")
    common_params = {
        "batch_size": hyper_parameters.batch_size,
        "pin_memory": torch.cuda.is_available() and hyper_parameters.data_config.pin_memory,
        "num_workers": 0 if platform.system() == "Darwin" else mp.cpu_count(),
        "collate_fn": collate_fn,
    }
    train_loader: DataLoaderType = DataLoader(
        dataset=train_ds,
        shuffle=True,
        drop_last=True,
        **common_params,
    )
    valid_loader: DataLoaderType = DataLoader(
        dataset=valid_ds,
        shuffle=False,
        drop_last=False,
        **common_params,
    )
    return train_loader, valid_loader


def get_dataset(
    hyper_parameters: config.HyperParameters,
    vision_transform: transforms.Compose,
) -> tuple[DataLoaderType, DataLoaderType]:
    logger.info("Loading Dataset...")
    dataset: datasets.Dataset = datasets.load_dataset(
        hyper_parameters.data_config.dataset,
        split="train",
        download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
    )  # type: ignore
    logger.info("Splitting dataset")
    train_test_dataset = dataset.train_test_split(
        seed=42, test_size=hyper_parameters.data_config.test_size
    )

    collate_fn = CollateFn(vision_transform)
    return _get_dataloaders(
        train_ds=train_test_dataset["train"],
        valid_ds=train_test_dataset["test"],
        hyper_parameters=hyper_parameters,
        collate_fn=collate_fn,
    )
