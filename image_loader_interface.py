from typing import Optional, Tuple

import numpy as np
from PIL import Image


class ImageLoaderInterface:
    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        ...

    def get_target_size(self) -> Tuple[int, int]:
        ...

    def _make_cache_key(self, args, *kwargs):
        ...

    def load_cached_image(self, cache_key: str) -> Optional[np.ndarray]:
        ...

    def save_to_cache(self, image: Image.Image, cache_key: str) -> None:
        ...
