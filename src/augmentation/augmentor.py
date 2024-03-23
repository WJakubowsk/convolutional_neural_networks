import numpy as np
import cv2
import random
from typing import List


def randomize(func):
    def wrapper(*args, **kwargs):
        if random.random() < 0.2:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


class Augmentor:
    def __init__(self):
        super().__init__()

    @randomize
    def _rotate_image(self, image: np.array, angle: int = 45) -> np.array:
        """Rotates the image by specified number of degrees clockwise."""
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))

    @randomize
    def _flip_image(self, image: np.array) -> np.array:
        """Flips the image horizontally clockwise."""
        return np.fliplr(image)

    @randomize
    def _sharpen_image(self, image: np.array) -> np.array:
        """Sharpens the image using a 3x3 kernel."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    @randomize
    def _blur_image(self, image: np.array) -> np.array:
        """Blurs the image using a 5x5 Gaussian kernel."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    @randomize
    def _brighten_image(self, image: np.array) -> np.array:
        """Increases the brightness of the image by 50%."""
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    @randomize
    def _translate_image(self, image: np.array, x: int = 0.1, y: int = 0.1) -> np.array:
        """Translates the image by a specified number of pixels in the x and y directions."""
        rows, cols = image.shape
        M = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(image, M, (cols, rows))

    def augment_data(self, image: np.array) -> List[np.array]:
        """
        Applies a series of augmentations to the input image, each with a 50% probability of being applied.
        Returns the list of obtained augmented images.
        """
        augmentations = [
            image,
            self._rotate_image(image),
            self._flip_image(image),
            self._sharpen_image(image),
            self._blur_image(image),
            self._brighten_image(image),
            self._translate_image(image),
        ]
        return list(filter(lambda x: x is not None, augmentations))
