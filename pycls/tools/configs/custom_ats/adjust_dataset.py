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
    for img_name, flag, img_path in dataset:
        if flag == "true":
            ture_list.append([img_name, flag, img_path])
        elif flag == "false":
            false_list.append([img_name, flag, img_path])

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

    dataset = []
    flags = set(["true", "false"])
    print("TODO:", sub_dir.as_posix())
    for img_path in sub_dir.glob("**/*.png"):
        img_name = img_path.stem
        flag = guess_flag(img_path.parts, flags)
        dataset.append([img_name, flag, img_path.as_posix()])

    test_data = defaultdict(list)
    for img_name, flag, _ in dataset:
        test_data[flag].append(img_name)
    print("[all] true/false:", len(test_data["true"]), len(test_data["false"]))
    print("[uni] true/false:", len(set(test_data["true"])), len(set(test_data["false"])))
    print("[chk] true&false:", set(test_data["true"]) & set(test_data["false"]))

    return split_dataset(dataset, rate, limit, seed)


def cache_dataset(output_dir, data_train, data_val):
    shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir + "/train/true", exist_ok=True)
    os.makedirs(output_dir + "/train/false", exist_ok=True)
    for img_name, flag, img_path in tqdm(data_train):
        out_file = "{}/train/{}/{}.png".format(output_dir, flag, img_name)
        shutil.copyfile(img_path, out_file)

    os.makedirs(output_dir + "/val/true", exist_ok=True)
    os.makedirs(output_dir + "/val/false", exist_ok=True)
    for img_name, flag, img_path in tqdm(data_val):
        out_file = "{}/val/{}/{}.png".format(output_dir, flag, img_name)
        shutil.copyfile(img_path, out_file)


def do_adjust_dataset(data_root, rate=0.5, limit=(10, 10000), seed=123):
    output_dir = "{}_split_{}".format(data_root, seed)

    data_train, data_val = [], []
    for sub_dir in sorted(Path(data_root).glob("*")):
        if not sub_dir.is_dir():
            continue

        train_, val_ = parse_sub_dir(sub_dir, rate, limit, seed)
        data_train.extend(train_)
        data_val.extend(val_)

    print("TODO:", "ALL")
    test_data = defaultdict(list)
    for img_name, flag, _ in (data_train + data_val):
        test_data[flag].append(img_name)
    print("[all] true/false:", len(test_data["true"]), len(test_data["false"]))
    print("[uni] true/false:", len(set(test_data["true"])), len(set(test_data["false"])))
    print("[chk] true&false:", set(test_data["true"]) & set(test_data["false"]))

    print("Tain:", len(data_train), ", Val:", len(data_val))
    cache_dataset(output_dir, data_train, data_val)
    return output_dir, data_train, data_val


if __name__ == "__main__":
    """
    ats_data_0000/
    ├── xxx_batch_0001
    │   ├── false
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── true
    │       ├── 3.png
    │       └── 4.png
    └── xxx_batch_0002
        ├── false
        │   ├── 5.png
        │   └── 6.png
        └── true
            ├── 7.png
            └── 8.png
    """
    data_root = "/mnt/d/work/tmp/ats/results/task_seed_1"
    print(do_adjust_dataset(data_root, rate=0.0, limit=(10, 100000), seed=1)[0])
