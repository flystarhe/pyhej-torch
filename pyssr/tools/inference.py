import os
import time

import cv2 as cv
import numpy as np
import pyssr.core.benchmark as benchmark
import pyssr.core.builders as builders
import pyssr.core.checkpoint as checkpoint
import pyssr.core.config as config
import pyssr.core.distributed as dist
import pyssr.core.logging as logging
import pyssr.core.meters as meters
import pyssr.core.net as net
import pyssr.core.optimizer as optim
import pyssr.datasets.loader as loader
import torch
from pyssr.core.config import cfg


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


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
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
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # Set complexity function to be module's complexity function
        model.complexity = model.module.complexity
    return model


def tensor2im(im_tensor):
    im_numpy = im_tensor.numpy()
    if im_numpy.shape[0] == 1:
        im_numpy = np.tile(im_numpy, (3, 1, 1))
    im_numpy = np.transpose(im_numpy, (1, 2, 0))
    im_numpy = (im_numpy * 0.5 + 0.5) * 255
    im_numpy = np.clip(im_numpy, 0, 255)
    return im_numpy.astype("uint8")


def keep_images(task_name, inputs, labels, preds, cur_epoch, cur_iter):
    names = ["{:03d}_{:03d}_{:03d}.png".format(cur_epoch, cur_iter, i) for i in range(inputs.size(0))]
    inputs, labels, preds = inputs.detach().cpu(), labels.detach().cpu(), preds.detach().cpu()
    out_dir = os.path.join(cfg.OUT_DIR, "{}_images".format(task_name))
    os.makedirs(out_dir, exist_ok=True)
    for name, _, b, c in zip(names, inputs, labels, preds):
        im_b, im_c = tensor2im(b), tensor2im(c)
        im_d = tensor2im(torch.abs(b - c))
        text = "{:.0f} {:.0f} {:.0f}".format(im_d.mean(), im_d.max(), im_d.var())
        if cfg.MODEL.VIS_THRESHOLD > 0:
            im_d = np.where(im_d <= 127 + cfg.MODEL.VIS_THRESHOLD, 0, im_d)
        cv.putText(im_d, text, (30, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
        rows = (np.concatenate((im_b, im_c), axis=1), np.concatenate((im_c, im_d), axis=1))
        cv.imwrite(os.path.join(out_dir, name), np.concatenate(rows, axis=0))


@torch.no_grad()
def test(cur_epoch=0):
    task_name = time.strftime("%m%d_%H%M_%S")
    os.makedirs(os.path.join(cfg.OUT_DIR, task_name))
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        err_mean, err_var = meters.compute_error(preds, labels)
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        err_mean, err_var = dist.scaled_all_reduce([err_mean, err_var])
        # Copy the errors from GPU to CPU (sync point)
        err_mean, err_var = err_mean.item(), err_var.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(err_mean, err_var, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
        # Keep predictions and targets
        keep_images(task_name, inputs, labels, preds, cur_epoch, cur_iter)
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    # Log output dir
    logger.info(os.path.join(cfg.OUT_DIR, task_name))
    return os.path.join(cfg.OUT_DIR, task_name)


def main():
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    test()


if __name__ == "__main__":
    main()
