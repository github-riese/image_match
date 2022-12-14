import random
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


def dataloader_wrapper(dataset: Dataset, randomize: bool, batch_size: int):
    total_items = len(dataset)
    indexes = list(range(total_items))

    index = total_items
    while True:
        if index >= total_items:
            index = 0
            if randomize:
                random.shuffle(indexes)
        batch_x = batch_y = None
        for b in range(min(total_items - index, batch_size)):
            index += 1
            x, y = dataset.__getitem__(index)
            x = normalize_x(x)
            batch_x = _add_tensor_to_tensors(batch_x, x)
            batch_y = _add_tensor_to_tensors(batch_y, y)
        yield batch_x, batch_y


def normalize_x(x: np.ndarray):
    x -= x.mean()
    return x / (x.std() + 1e-7)


def de_normalize_x(x: np.ndarray):
    return (x - x.min())/2


def _add_tensor_to_tensors(batch: Optional[np.ndarray], tensor: np.ndarray) -> np.ndarray:
    if batch is None:
        batch = np.zeros((0, tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
    batch = np.concatenate([batch, tensor.reshape((1, 80, 80, 3))], axis=0)
    return batch
