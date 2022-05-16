from typing import Optional

import numpy as np
import torch.utils.data
from pandas import DataFrame

from image_loader_interface import ImageLoaderInterface


class Dataset(torch.utils.data.Dataset):
    _data: DataFrame
    _item_count: int
    _reserved_count: int
    _image_loader: ImageLoaderInterface
    _transpose_for_torch: bool

    def __init__(self, data: DataFrame, reserve_percent: Optional[float] = None,
                 image_loader: ImageLoaderInterface = None, transpose_for_torch: bool = True):
        super(Dataset, self).__init__()
        self._data = data
        count = data.__len__()
        usable_count = count if reserve_percent is None else count - int(count * reserve_percent / 100)
        self._item_count = usable_count
        self._reserved_count = count - usable_count
        self._image_loader = image_loader
        self._transpose_for_torch = transpose_for_torch

    def get_reserved_data(self):
        return Dataset(self._data[self._item_count:], image_loader=self._image_loader)

    def __len__(self):
        return self._item_count

    def __getitem__(self, item):
        row = self._data.iloc[item]
        box = row['x1'], row['y1'], row['x2'], row['y2']
        image = self._image_loader.load_image(row['path'], 'jewellery' if row['type'] != 'junk' else None)
        image = np.array(image, dtype=np.float32)
        image /= 255
        if self._transpose_for_torch:
            image = image.transpose((2, 0, 1))
        return image, np.array(box, dtype=np.float32)
