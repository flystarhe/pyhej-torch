import numpy as np


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")


def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def horizontal_flip(im, p, order="CHW"):
    """Performs horizontal flip (CHW or HWC format)."""
    assert order in ["CHW", "HWC"]
    if np.random.uniform() < p:
        if order == "CHW":
            im = im[:, :, ::-1]
        else:
            im = im[:, ::-1, :]
    return im


def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y: (y + size), x: (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop


def test_crop(im, size):
    """Performs random crop (CHW format)."""
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    return [x, y, x + size, y + size]


def crop_image(im, xyxy):
    """Performs crop image (CHW format)."""
    x1, y1, x2, y2 = xyxy
    im_crop = im[:, y1: y2, x1: x2]
    return im_crop


def scale_down(im, scale_factor):
    """Performs scale down (CHW format)."""
    assert scale_factor == 2, "scale_factor must be 2"
    im_scale = im[:, ::2, ::2]
    return im_scale
