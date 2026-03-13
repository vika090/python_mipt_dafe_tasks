import numpy as np


def get_dominant_color_info(
    image: np.ndarray,
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    if threshold < 1:
        raise ValueError("threshold must be positive")

    pixels = image.flatten()

    groups = np.zeros(256, dtype=np.int64)

    for i in pixels:
        groups[i] += 1

    max_size = 0
    max_group = 0

    for color in range(256):
        left = max(0, color - threshold + 1)
        right = min(255, color + threshold - 1)

        cur_group = np.sum(groups[left : right + 1])

        if cur_group > max_size:
            max_size = cur_group
            max_group = color

    dominate_color = np.uint8(max_group)
    persent = max_size / pixels.size * 100

    return dominate_color, persent