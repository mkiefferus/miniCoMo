from .mnist import MNISTDataset

from typing import Any

from torch.utils.data import DataLoader


def get_data_loader(dataset: Any,
                    batch_size: int,
                    num_workers: int = 8,
                    shuffle: bool = True,
                    drop_last: bool = True):
    if isinstance(dataset, str):
        dataset = eval(dataset)()

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=num_workers,
                      pin_memory=True)