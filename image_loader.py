from os import PathLike
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from image_loader_interface import ImageLoaderInterface


class ImageLoader(ImageLoaderInterface):
    _paths: Dict[str, PathLike]
    _target_size: Optional[Tuple[int, int]]

    def __init__(self, base_paths: Dict[str, PathLike], image_size: Optional[Tuple[int, int]]):
        super(ImageLoader, self).__init__()
        self._paths = base_paths
        self._target_size = image_size

    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if img_type in self._paths:
            the_path = self._paths[img_type] + f"/{path}"
        else:
            the_path = path
        image = Image.open(the_path)
        image = self._resize(image, desired_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.asarray(image)

    def _resize(self, image: Image.Image, override: Optional[Tuple[int, int]]):
        if override is not None:
            size = override
        elif self._target_size is not None:
            size = self._target_size
        else:
            return image
        return image.resize(size)
