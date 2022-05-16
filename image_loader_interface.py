from typing import Optional, Tuple

import numpy as np


class ImageLoaderInterface:
    def load_image(self, path: str, img_type: str = None, desired_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        ...
