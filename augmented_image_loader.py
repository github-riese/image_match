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
        image = self.load_image(path, img_type, desired_size)
        if bbox is None:
            return image
        coords = self._to_coords(bbox, (image.shape[0], image.shape[1]))
        image = Image.fromarray(image)
        image = image.crop(coords)
        image = image.resize(self._output_size if desired_size is None else desired_size)
        return np.array(image)

    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        return self._image_loader.load_image(path, img_type, desired_size)

    @staticmethod
    def _to_coords(bbox, size):
        l, t, r, b = int(bbox[0] * size[0]), int(bbox[1] * size[1]), int(bbox[2] * size[0]), int(bbox[3] * size[1])
        if l > r:
            (r, l) = (l, r)
        if t > b:
            (t, b) = (b, t)
        return l, t, r, b
