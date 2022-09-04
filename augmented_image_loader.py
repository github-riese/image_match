from typing import Optional, Tuple

from PIL import Image
import numpy as np

from image_loader import ImageLoader
from image_loader_interface import ImageLoaderInterface


class AugmentedImageLoader(ImageLoaderInterface):
    _image_loader: ImageLoader
    _output_size: Tuple[int, int]

    def __init__(self, image_loader: ImageLoader, size: Tuple[int, int]):
        self._image_loader = image_loader
        self._output_size = size

    def load_augmented_image(self, path: str, img_type: str = None,
                             desired_size: Optional[Tuple[int, int]] = None,
                             bbox: Optional[Tuple[float, float, float, float]] = None):
        out_size = self._output_size if desired_size is None else desired_size
        cache_key = self._make_cache_key(path, out_size, bbox)
        cached = self._image_loader.load_cached_image(cache_key)
        if cached is not None:
            return cached
        image = self._image_loader.load_image(path, img_type, desired_size)

        if bbox is None:
            return image
        coords = self._to_coords(bbox, (image.shape[0], image.shape[1]))
        image = Image.fromarray(image)
        image = image.crop(coords)
        image = image.resize(out_size)
        self._image_loader.save_to_cache(image, cache_key)
        return np.array(image)

    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None,
                   cache_key: str = None) -> Optional[np.ndarray]:
        cached = self._image_loader.load_cached_image(cache_key)
        if cached is not None:
            return cached
        return self._image_loader.load_image(path, img_type, desired_size)

    def _make_cache_key(self, path: str, desired_size: Optional[Tuple[int, int]] = None,
                        bbox: Optional[Tuple[float, float, float, float]] = None):
        parts = path.split('/')
        name = parts[-1].split('.')
        ext = name[-1]
        name = '_'.join(name[:-1])
        size_part = map(str, desired_size if desired_size is not None else self._image_loader.get_target_size())
        bbox_part = map(str, bbox if bbox is not None else ['no_box'])
        return name + '_' + '_'.join(parts[:-1]) + '_' + '_'.join(size_part) + '_' + '_'.join(bbox_part) + '.' + ext

    @staticmethod
    def _to_coords(bbox, size):
        l, t, r, b = int(bbox[0] * size[0]), int(bbox[1] * size[1]), int(bbox[2] * size[0]), int(bbox[3] * size[1])
        if l > r:
            (r, l) = (l, r)
        if t > b:
            (t, b) = (b, t)
        return l, t, r, b
