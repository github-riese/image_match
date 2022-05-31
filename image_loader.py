import os.path
from os import PathLike
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from image_loader_interface import ImageLoaderInterface


class ImageLoader(ImageLoaderInterface):
    _paths: Dict[str, PathLike]
    _target_size: Optional[Tuple[int, int]]
    _cache_path: Optional[str]

    def __init__(self, base_paths: Dict[str, PathLike], image_size: Optional[Tuple[int, int]],
                 cache_path: Optional[str]):
        super(ImageLoader, self).__init__()
        self._paths = base_paths
        self._target_size = image_size
        self._cache_path = cache_path

    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if img_type in self._paths:
            the_path = self._paths[img_type] + f"/{path}"
        else:
            the_path = path
        cache_key = self._make_cache_key(the_path, desired_size)
        image = self.load_cached_image(cache_key)
        if image is not None:
            return image

        image = Image.open(the_path)
        image = self._resize(image, desired_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        self.save_to_cache(image, cache_key)
        return np.asarray(image)

    def get_target_size(self) -> Tuple[int, int]:
        return self._target_size

    def _resize(self, image: Image.Image, override: Optional[Tuple[int, int]]):
        if override is not None:
            size = override
        elif self._target_size is not None:
            size = self._target_size
        else:
            return image
        return image.resize(size)

    def _make_cache_key(self, path: str, size: Optional[Tuple[int, int]] = None) -> str:
        parts = path.split('/')
        name = parts[-1].split('.')
        ext = name[-1]
        name = '_'.join(name[:-1])
        size_part = map(str, size if size is not None else self._target_size)
        return name + '_' + '_'.join(parts[:-1]) + '_' + '_'.join(size_part) + '.' + ext

    def save_to_cache(self, image: Image.Image, cache_key: str) -> None:
        if self._cache_path is None:
            return
        if not os.path.isdir(self._cache_path):
            return

        path = self._cache_path + f"/{cache_key}"
        image.save(path)

    def load_cached_image(self, cache_key: str) -> Optional[np.ndarray]:
        if self._cache_path is None:
            return None
        path = self._cache_path + f"/{cache_key}"
        if not os.path.isfile(path):
            return None
        return np.asarray(Image.open(path))
