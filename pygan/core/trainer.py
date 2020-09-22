import os

import numpy as np
import pygan.core.builders as builders
import pygan.core.config as config
import pygan.core.distributed as dist
import pygan.core.logging as logging
import pygan.core.meters as meters
import pygan.core.net as net
import pygan.core.optimizer as optim
import pygan.datasets.loader as loader
import torch
from pygan.core.config import cfg


from pygan.module_dcgan.model import Img2Img
builders.register_model("img2img", Img2Img)


logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model(is_train):
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.get_model()(is_train)
    logger.info("Model:\n{}".format(model))
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = model.parallel(device=cur_device)
        # Set complexity function to be module's complexity function
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    model.update_lr(lr, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda()
        model.set_input((inputs, labels))
        model.optimize_parameters()
        # Compute the errors
        errs, keys = model.get_current_losses()
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        errs = dist.scaled_all_reduce(errs)
        train_meter.iter_toc()
        # Update and log stats
        kwargs = {k: e.item() for k, e in zip(keys, errs)}
        train_meter.update_stats(lr, **kwargs)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    criterion = torch.nn.L1Loss().cuda()
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda()
        model.set_input((inputs, labels))
        model.forward()
        # Compute the errors
        err_l1 = criterion(model.fake, model.peak)
        errs, keys = [err_l1], ["loss_G_idt"]
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        errs = dist.scaled_all_reduce(errs)
        test_meter.iter_toc()
        # Update and log stats
        kwargs = {k: e.item() for k, e in zip(keys, errs)}
        test_meter.update_stats(None, **kwargs)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model(is_train=True)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = model.load_networks(last_checkpoint)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        model.load_networks(cfg.TRAIN.WEIGHTS)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.ComplexityMeter("train", len(train_loader))
    test_meter = meters.ComplexityMeter("test", len(test_loader))
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, train_meter, cur_epoch)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = model.save_networks(cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model(is_train=False)
    # Load model weights
    model.load_networks(cfg.TEST.WEIGHTS)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.ComplexityMeter("test", len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)
