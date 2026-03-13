import numpy as np


def get_dominant_color_info(
    image: np.ndarray,
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    if threshold < 1:
        raise ValueError("threshold must be positive")

    pixels = image.flatten()

    groups = np.zeros(256, dtype=np.int64)

    for color in range(256):
        groups[color] = np.sum(pixels == color)

    max_size = 0
    max_group = 0

    for color in range(256):
        if groups[color] != 0:
            low = max(0, color - (threshold - 1))
            high = min(255, color + (threshold - 1))

            cur_group = np.sum(groups[low : high + 1])

            if cur_group > max_size:
                max_size = cur_group
                max_group = color

    dominate_color = np.uint8(max_group)
    percent = (max_size / pixels.size) * 100

    return dominate_color, percent
