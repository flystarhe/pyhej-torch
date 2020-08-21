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


def parse_sub_dir(sub_dir):
    dataset = defaultdict(dict)
    flags = set(["true", "false"])
    for img_path in sorted(sub_dir.glob("**/*.png")):
        img_name = img_path.stem
        flag = guess_flag(img_path.parts, flags)
        if img_name.endswith("_red"):
            img_tag = img_name[:-4] + "/" + flag
            dataset[img_tag]["red"] = img_path.as_posix()
        elif img_name.endswith("_blue"):
            img_tag = img_name[:-5] + "/" + flag
            dataset[img_tag]["blue"] = img_path.as_posix()
        else:
            print("Failed:", img_path)

    outputs = []
    logs = ["\nmiss data:"]
    for img_tag, v in dataset.items():
        if "red" in v and "blue" in v:
            outputs.append([img_tag, v["red"], v["blue"]])
        else:
            logs.append("{} - {} - {}".format(len(logs), img_tag, v))
    print(sub_dir.as_posix(), ":", len(outputs), "/", len(dataset), "\n".join(logs), "\n")
    return outputs


def keep_dataset(output_dir, dataset):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir / "true", exist_ok=True)
    os.makedirs(output_dir / "false", exist_ok=True)
    for img_tag, img_red, img_blue in tqdm(dataset):
        img_red = cv.imread(img_red, 0)
        img_blue = cv.imread(img_blue, 0)
        img_name, flag = img_tag.split("/")
        img = np.stack([img_blue, img_blue, img_red], axis=2)
        out_file = "{}/{}/{}.png".format(output_dir, flag, img_name)
        cv.imwrite(out_file, img)


def do_build_dataset(data_root, output_dir):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir = output_dir / data_root.name

    dataset = []
    for sub_dir in sorted(data_root.glob("*")):
        if sub_dir.is_dir():
            dataset.extend(parse_sub_dir(sub_dir))

    test_data = defaultdict(set)
    for img_tag, _, _ in dataset:
        img_name, flag = img_tag.split("/")
        test_data[flag].add(img_name)

    print("[count] true: {}, false: {}".format(len(test_data["true"]), len(test_data["false"])))
    print("[count] true&false: {}".format(len(test_data["true"] & test_data["false"])))
    print("saved images:", len(test_data["true"]) + len(test_data["false"]))

    keep_dataset(output_dir, dataset)
    return output_dir


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
    data_root = "/mnt/f/ats/data/batch_0806"
    output_dir = "/mnt/f/ats/results/data_0821"
    print(do_build_dataset(data_root, output_dir))
