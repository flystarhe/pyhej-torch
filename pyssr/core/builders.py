import torch
from pyssr.core.config import cfg
from pyssr.models.subpixel import SubPixel
from pyssr.models.unet import UNet


# Supported models
_models = {"subpixel": SubPixel, "unet": UNet}

# Supported loss functions
_loss_funs = {"mse": torch.nn.MSELoss, "l1": torch.nn.L1Loss}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
