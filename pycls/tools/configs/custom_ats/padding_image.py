import cv2 as cv
import numpy as np
import shutil

from pathlib import Path
from tqdm import tqdm


def padding_image(image, target_size=64):
    image = image[:target_size, :target_size]

    h, w, c = image.shape
    new_shape = (target_size, target_size, c)
    padded = np.zeros(new_shape, dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def padding_dataset(data_root, target_size=64):
    output_dir = (data_root + "_padded")
    shutil.rmtree(output_dir, ignore_errors=True)

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    img_list = list(data_root.glob("**/*.png"))

    for img_path in tqdm(img_list):
        image = cv.imread(img_path.as_posix(), 1)
        padded = padding_image(image, target_size)
        out_file = output_dir / img_path.relative_to(data_root)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(out_file.as_posix(), padded)
    return data_root, output_dir


if __name__ == "__main__":
    data_root = "/mnt/f/ats/results/data_0821_split_100"
    print(padding_dataset(data_root, target_size=64))
