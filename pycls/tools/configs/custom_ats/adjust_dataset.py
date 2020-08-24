import cv2 as cv
import numpy as np
import os
import shutil

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def guess_flag(path_parts, flags):
    for flag in path_parts[::-1]:
        flag = flag.lower()
        if flag in flags:
            return flag
    return "none"


def agent_sampling(data, rate=0.5, seed=123, limit=(10, 100000)):
    data = sorted(data, key=lambda x: x[0])

    np.random.seed(seed)
    np.random.shuffle(data)

    rate = max(0, min(0.9, rate))
    n_train = int(len(data) * rate)

    a, b = limit
    n_train = max(a, min(b, n_train))

    return data[:n_train], data[n_train:]


def split_dataset(dataset, flags, rate=0.5, seed=123, limit=(10, 100000)):
    groups = defaultdict(list)
    for img_name, flag, img_path in dataset:
        groups[flag].append([img_name, flag, img_path])

    data_train, data_val = [], []
    for flag, vals in groups.items():
        if flag not in flags:
            print("[skip] num:", flag, "=", len(vals))
            continue
        _train, _val = agent_sampling(vals, rate, limit, seed)
        data_train.extend(_train)
        data_val.extend(_val)
    return data_train, data_val


def parse_sub_dir(sub_dir, flags):
    dataset = []
    for img_path in sorted(Path(sub_dir).glob("**/*.png")):
        img_name = img_path.stem
        flag = guess_flag(img_path.parts, flags)
        dataset.append([img_name, flag, img_path.as_posix()])
    return dataset


def clean_dataset(test_dataset, data_train, data_val):
    _best = dict()
    for img_name, _, img_path in test_dataset:
        _best[img_name] = img_path
    _best = set(_best.values())

    res_train, rep_train = [], []
    for img_name, flag, img_path in data_train:
        if img_path in _best:
            res_train.append([img_name, flag, img_path])
        else:
            rep_train.append([img_name, flag, img_path])

    res_val, rep_val = [], []
    for img_name, flag, img_path in data_val:
        if img_path in _best:
            res_val.append([img_name, flag, img_path])
        else:
            rep_val.append([img_name, flag, img_path])

    print("\n[clean] replace_train:", len(rep_train), ", replace_val:", len(rep_val))
    return res_train, res_val


def keep_dataset(output_dir, flags, data_train, data_val):
    shutil.rmtree(output_dir, ignore_errors=True)

    for flag in flags:
        os.makedirs("{}/train/{}".format(output_dir, flag), exist_ok=True)
        os.makedirs("{}/val/{}".format(output_dir, flag), exist_ok=True)

    for img_name, flag, img_path in tqdm(data_train):
        out_file = "{}/train/{}/{}.png".format(output_dir, flag, img_name)
        shutil.copyfile(img_path, out_file)

    for img_name, flag, img_path in tqdm(data_val):
        out_file = "{}/val/{}/{}.png".format(output_dir, flag, img_name)
        shutil.copyfile(img_path, out_file)
    print("The save path:", output_dir)
    return output_dir


def do_adjust_dataset(data_root, flags, rate=0.5, seed=123, limit=(10, 100000)):
    test_dataset, data_train, data_val = [], [], []
    for sub_dir in sorted(Path(data_root).glob("*")):
        if sub_dir.is_dir():
            print("TODO:", sub_dir.as_posix())
            dataset = parse_sub_dir(sub_dir.as_posix(), flags)
            _train, _val = split_dataset(dataset, flags, rate, limit, seed)
            test_dataset.extend(dataset)
            data_train.extend(_train)
            data_val.extend(_val)
    data_train, data_val = clean_dataset(test_dataset, data_train, data_val)
    print(len(test_dataset), ", train:", len(data_train), ", val:", len(data_val))
    output_dir = "{}_split_r{}_s{}".format(data_root, int(rate * 100), seed)
    return keep_dataset(output_dir, flags, data_train, data_val)


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
    flags = set(["true", "false"])
    data_root = "/mnt/f/ats/results/data_0821"
    do_adjust_dataset(data_root, flags, rate=0.5, seed=123, limit=(10, 100000))
