import multiprocessing as mp
from typing import Callable

import datasets
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader

from src import config


def collate_fn(batch: list[dict[str, Image.Image | str]]) -> dict[str, list[Image.Image | str]]:
    return {
        "images": [data_instance["image"] for data_instance in batch],
        "texts": [data_instance["caption"] for data_instance in batch],
    }


def _get_dataloaders(
    train_ds: datasets.Dataset,
    valid_ds: datasets.Dataset,
    hyper_parameters: config.HyperParameters,
    collate_fn: Callable,
) -> tuple[DataLoader, DataLoader]:
    logger.info("Creating Dataloaders")
    common_params = {
        "batch_size": hyper_parameters.batch_size,
        "pin_memory": True,
        "num_workers": mp.cpu_count(),
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common_params,
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        drop_last=False,
        **common_params,
    )
    return train_loader, valid_loader


def get_dataset(
    hyper_parameters: config.HyperParameters,
) -> tuple[DataLoader, DataLoader]:
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

    return _get_dataloaders(
        train_ds=train_test_dataset["train"],
        valid_ds=train_test_dataset["test"],
        hyper_parameters=hyper_parameters,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # do not want to do these imports in general
    import os

    from tqdm.auto import tqdm

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hyper_parameters = config.HyperParameters()
    train_dl, valid_dl = get_dataset(hyper_parameters)

    batch = next(iter(train_dl))
    print({k: v.shape for k, v in batch.items()})  # torch.Size([1, 3, 128, 128])

    for batch in tqdm(train_dl):
        continue
