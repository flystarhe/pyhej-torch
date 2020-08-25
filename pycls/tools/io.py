import os
import json
import pickle
import shutil
from pathlib import Path


def mkdir(out_dir, is_file=False):
    out_dir = Path(out_dir)
    if is_file:
        out_dir = out_dir.parent

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    return True


def copy_img(img_path, out_dir):
    mkdir(out_dir, is_file=False)
    out_file = Path(out_dir) / Path(img_path).name
    shutil.copyfile(img_path, out_file)
    return img_path


def load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_file):
    mkdir(pkl_file, is_file=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)
    return pkl_file


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    mkdir(json_file, is_file=True)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def load_csv(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    return lines


def save_csv(lines, csv_file):
    mkdir(csv_file, is_file=True)
    with open(csv_file, "w") as f:
        f.write("\n".join(lines))
    return csv_file
