from typing import Optional

import numpy as np
import torch.utils.data
from pandas import DataFrame

from augmented_image_loader import AugmentedImageLoader
from image_loader_interface import ImageLoaderInterface


class Dataset(torch.utils.data.Dataset):
    _data: DataFrame
    _item_count: int
    _reserved_count: int
    _image_loader: AugmentedImageLoader

    def __init__(self, data: DataFrame, reserve_percent: Optional[float] = None,
                 image_loader: ImageLoaderInterface = None):
        super(Dataset, self).__init__()
        self._data = data
        self._data.sort_values(by='product_id')
        count = data.__len__()
        usable_count = count if reserve_percent is None else count - int(count * reserve_percent / 100)
        self._item_count = usable_count
        self._reserved_count = count - usable_count
        if isinstance(image_loader, AugmentedImageLoader):
            self._image_loader = image_loader
        else:
            raise TypeError("ImageLoader must be AugmentedImageLoader for this kind of dataset")

    def get_reserved_data(self):
        return Dataset(self._data[self._item_count:], image_loader=self._image_loader)

    def __len__(self):
        return self._item_count

    def __getitem__(self, item):
        row = self._data.iloc[item]
        image = self._image_loader.load_augmented_image(row['path'], 'jewellery' if row['type'] != 'junk' else None,
                                                        bbox=self._get_bbox_from_df(row))
        if row['flavour'] != 'original':
            product_id = row['product_id']
            original_image_record = self._data.loc[
                self._data['product_id'].str.contains(product_id) & self._data['flavour'].str.contains('original')]

            if len(original_image_record['path']) != 1:
                print(original_image_record)
            orig_image = self._image_loader.load_augmented_image(original_image_record['path'].item(), 'jewellery',
                                                                 bbox=self._get_bbox_from_df(original_image_record))
        else:
            orig_image = image
        image = np.array(image, dtype=np.float32)
        image /= 255
        orig_image = np.array(orig_image, dtype=np.float32) / 255
        shape = image.shape
        return image, orig_image
        # image.reshape((1, shape[0], shape[1], shape[2])), orig_image.reshape((1, shape[0], shape[1], shape[2]))

    @staticmethod
    def _get_bbox(row):
        return row['x1'], row['y1'], row['x2'], row['y2']

    @staticmethod
    def _get_bbox_from_df(row: DataFrame):
        x1 = row['x1'].item()
        y1 = row['y1'].item()
        x2 = row['x2'].item()
        y2 = row['y2'].item()
        return x1, y1, x2, y2
