import argparse
import os
import sys
from yacs.config import CfgNode as CfgNode


# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C


# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = "img2img"
_C.MODEL.G = "resnet"
_C.MODEL.D = "nlayer"

# Number of input image channels
_C.MODEL.NC = 1


# ------------------------------------------------------------------------------------ #
# RESNET options
# ------------------------------------------------------------------------------------ #
_C.RESNET = CfgNode()
_C.RESNET.NGF = 64
_C.RESNET.NORM_LAYER = "instance"
_C.RESNET.USE_DROPOUT = False
_C.RESNET.N_BLOCKS = 6
_C.RESNET.PADDING_TYPE = "reflect"
_C.RESNET.NO_ANTIALIAS = False
_C.RESNET.NO_ANTIALIAS_UP = False


# ------------------------------------------------------------------------------------ #
# NLAYER options
# ------------------------------------------------------------------------------------ #
_C.NLAYER = CfgNode()
_C.NLAYER.NDF = 64
_C.NLAYER.N_LAYERS = 3
_C.NLAYER.NORM_LAYER = "instance"
_C.NLAYER.NO_ANTIALIAS = False


# ------------------------------------------------------------------------------------ #
# DCGAN options
# ------------------------------------------------------------------------------------ #
_C.DCGAN = CfgNode()
_C.DCGAN.NGF = 64
_C.DCGAN.NDF = 64


# ------------------------------------------------------------------------------------ #
# Optimizer options
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0


# ------------------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Image size
_C.TRAIN.IM_SIZE = 224

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""


# ------------------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Image size
_C.TEST.IM_SIZE = 256

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ------------------------------------------------------------------------------------ #
# CUDNN options
# ------------------------------------------------------------------------------------ #
_C.CUDNN = CfgNode()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used
_C.CUDNN.BENCHMARK = True


# ------------------------------------------------------------------------------------ #
# Misc options
# ------------------------------------------------------------------------------------ #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 100

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]


# ------------------------------------------------------------------------------------ #
# Default config
# ------------------------------------------------------------------------------------ #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "See pygan/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)
