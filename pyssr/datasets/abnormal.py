import os

import cv2 as cv
import numpy as np
import pyssr.core.logging as logging
import pyssr.datasets.transforms as transforms
import torch.utils.data
from lxml import etree
from pathlib import Path
from pyssr.core.config import cfg
from xml.etree import ElementTree


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [0.5, 0.5, 0.5]  # [0.406, 0.456, 0.485]
_SD = [0.5, 0.5, 0.5]  # [0.225, 0.224, 0.229]


def do_labelImg(xml_path):
    shapes = []
    if not os.path.isfile(xml_path):
        return shapes
    parser = etree.XMLParser(encoding="utf-8")
    xmltree = ElementTree.parse(xml_path, parser=parser).getroot()
    for object_iter in xmltree.findall("object"):
        bndbox = object_iter.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        shapes.append([xmin, ymin, xmax, ymax])
    return shapes


def check_crop(patch, boxes):
    if boxes.shape[0] == 0:
        return True

    patch = patch.astype(np.float32)
    boxes = boxes.astype(np.float32)

    x_beg = np.maximum(patch[0], boxes[:, 0])
    y_beg = np.maximum(patch[1], boxes[:, 1])
    x_end = np.minimum(patch[2], boxes[:, 2])
    y_end = np.minimum(patch[3], boxes[:, 3])
    overlap_w = np.maximum(x_end - x_beg + 1, 0)
    overlap_h = np.maximum(y_end - y_beg + 1, 0)

    inner = (overlap_w * overlap_h) > 0
    if np.any(inner):
        return False
    return True


class Abnormal(torch.utils.data.Dataset):
    """Abnormal dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for Abnormal".format(split)
        self._data_path, self._split = data_path, split
        self._construct_imdb()

        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.upscale_factor = cfg.SUBPIXEL.UPSCALE_FACTOR

    def _construct_imdb(self):
        self._imdb = []
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        for im_path in Path(split_path).glob("**/*.png"):
            gt_path = im_path.with_suffix(".xml")
            im_path = im_path.as_posix()
            gts = do_labelImg(gt_path)
            gts = np.array(gts, dtype=np.float32)
            self._imdb.append({"im_path": im_path, "gts": gts})
        logger.info("Number of images: {}".format(len(self._imdb)))

    def _prepare_im(self, im, gts, max_iter=10):
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # Crop the image for training / testing
        train_size = cfg.TRAIN.IM_SIZE
        if self._split == "train":
            im = transforms.zero_pad(im=im, pad_size=32)
            im = transforms.horizontal_flip(im=im, p=0.5)
            for _ in range(max_iter):
                xyxy = transforms.test_crop(im=im, size=train_size)
                if check_crop(xyxy, gts):
                    break
            im = transforms.crop_image(im=im, xyxy=xyxy)
        else:
            im = transforms.random_crop(im=im, size=cfg.TEST.IM_SIZE)
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        input = transforms.scale_down(im, self.upscale_factor).copy()
        return input, im

    def __getitem__(self, index):
        # Load the image
        if self.in_channels == 3:
            im = cv.imread(self._imdb[index]["im_path"], 1)
            im = im.astype(np.float32, copy=False)
        elif self.in_channels == 1:
            im = cv.imread(self._imdb[index]["im_path"], 0)
            im = im.astype(np.float32, copy=False)
            im = im[:, :, np.newaxis]
        else:
            raise Exception("in_channels must be 1 or 3")
        # Retrieve the gts
        gts = self._imdb[index]["gts"]
        # Prepare the image for training / testing
        input, target = self._prepare_im(im, gts)
        return input, target

    def __len__(self):
        return len(self._imdb)
