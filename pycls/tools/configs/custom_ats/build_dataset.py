import cv2 as cv
import numpy as np
import os
import shutil

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def padding_image(image, target_size=64):
    if isinstance(image, (str, Path)):
        image = cv.imread(str(image), 1)

    image = image[:target_size, :target_size]

    h, w, c = image.shape
    new_shape = (target_size, target_size, c)
    padded = np.zeros(new_shape, dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def agent_sampling(data, rate=0.5, limit=(10, 10000), seed=123):
    data = sorted(data, key=lambda x: x[0])

    np.random.seed(seed)
    np.random.shuffle(data)

    rate = max(0, min(0.9, rate))
    n_train = int(len(data) * rate)

    a, b = limit
    n_train = max(a, min(b, n_train))

    return data[:n_train], data[n_train:]


def split_dataset(dataset, rate=0.5, limit=(10, 10000), seed=123):
    ture_list, false_list = [], []
    for img_tag, img_red, img_blue in dataset:
        _, flag = img_tag.split("/")
        if flag == "true":
            ture_list.append([img_tag, img_red, img_blue])
        elif flag == "false":
            false_list.append([img_tag, img_red, img_blue])

    true_train, true_val = agent_sampling(ture_list, rate, limit, seed)
    false_train, false_val = agent_sampling(false_list, rate, limit, seed)
    return true_train + false_train, true_val + false_val


def parse_sub_dir(sub_dir, rate=0.5, limit=(10, 10000), seed=123):
    def guess_flag(path_parts, flags):
        for flag in path_parts[::-1]:
            flag = flag.lower()
            if flag in flags:
                return flag
        return "none"

    dataset = defaultdict(dict)
    flags = set(["true", "false"])
    for img_path in sub_dir.glob("**/*.png"):
        img_name = img_path.stem
        flag = guess_flag(img_path.parts, flags)
        if img_name.endswith("_red"):
            img_key = img_name[:-4] + "/" + flag
            img_cls = "red"
        elif img_name.endswith("_blue"):
            img_key = img_name[:-5] + "/" + flag
            img_cls = "blue"
        else:
            print("Failed:", img_path)
            continue
        dataset[img_key][img_cls] = img_path.as_posix()

    logs = []
    outputs = []
    for k, v in dataset.items():
        if "red" in v and "blue" in v:
            outputs.append([k, v["red"], v["blue"]])
        else:
            logs.append("{} - {} - {}".format(len(logs), k, v))
    print(sub_dir.as_posix(), ":", len(outputs), "/", len(dataset), "\n", "\n".join(logs))
    return split_dataset(outputs, rate, limit, seed)


def cache_dataset(output_dir, data_train, data_val):
    shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir + "/train/true", exist_ok=True)
    os.makedirs(output_dir + "/train/false", exist_ok=True)
    for img_tag, img_red, img_blue in tqdm(data_train):
        img_name, flag = img_tag.split("/")
        img_red = cv.imread(img_red, 0)
        img_blue = cv.imread(img_blue, 0)
        img = np.stack([img_blue, img_blue, img_red], axis=2)
        img = padding_image(img, target_size=64)
        out_file = "{}/train/{}/{}.png".format(output_dir, flag, img_name)
        cv.imwrite(out_file, img)

    os.makedirs(output_dir + "/val/true", exist_ok=True)
    os.makedirs(output_dir + "/val/false", exist_ok=True)
    for img_tag, img_red, img_blue in tqdm(data_val):
        img_name, flag = img_tag.split("/")
        img_red = cv.imread(img_red, 0)
        img_blue = cv.imread(img_blue, 0)
        img = np.stack([img_blue, img_blue, img_red], axis=2)
        img = padding_image(img, target_size=64)
        out_file = "{}/val/{}/{}.png".format(output_dir, flag, img_name)
        cv.imwrite(out_file, img)


def do_build_dataset(data_root, output_dir, rate=0.5, limit=(10, 10000), seed=123):
    data_train, data_val = [], []
    output_dir = "{}_{}".format(output_dir, seed)

    for sub_dir in sorted(Path(data_root).glob("*")):
        if not sub_dir.is_dir():
            continue

        train_, val_ = parse_sub_dir(sub_dir, rate, limit, seed)
        data_train.extend(train_)
        data_val.extend(val_)

    print("Tain:", len(data_train), ", Val:", len(data_val))
    cache_dataset(output_dir, data_train, data_val)
    return output_dir, data_train, data_val


if __name__ == "__main__":
    """
    ats_data_0000/
    ├── xxx_batch_0001
    │   ├── false
    │   │   ├── 1_blue.png
    │   │   ├── 1_red.png
    │   │   ├── 2_blue.png
    │   │   └── 2_red.png
    │   └── true
    │       ├── 3_blue.png
    │       ├── 3_red.png
    │       ├── 4_blue.png
    │       └── 4_red.png
    └── xxx_batch_0002
        ├── false
        │   ├── 5_blue.png
        │   ├── 5_red.png
        │   ├── 6_blue.png
        │   └── 6_red.png
        └── true
            ├── 7_blue.png
            ├── 7_red.png
            ├── 8_blue.png
            └── 8_red.png
    """
    data_root = "/mnt/d/work/tmp/ats/data/final_real_test_66k"
    output_dir = "/mnt/d/work/tmp/ats/results/final_real_test_66k"
    print(do_build_dataset(data_root, output_dir, rate=0.0, limit=(0, 100000), seed=1)[0])
