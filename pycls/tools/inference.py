import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.net as net
import pycls.datasets.loader as loader

from pycls.core.config import cfg


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


def search_thr(data, s1_thr=5, s2_thr=80, out_file=None):
    """Search from: [(label,pred_label,pred_score),]"""
    def test_thr(label, pred_label, pred_score):
        y_ = [b if a == 0 else 1 - b for a, b in zip(pred_label, pred_score)]
        y, y_ = np.array(label, dtype="float32"), np.array(y_, dtype="float32")

        x = np.linspace(0, 1, 100, endpoint=False)[1:]
        s1, s2 = np.zeros_like(x), np.zeros_like(x)
        for i, xi in enumerate(x):
            total_ture = np.sum(y > 0.5)
            total_false = np.sum(y < 0.5)
            num_fn = np.sum((y_ >= xi) * (y > 0.5))
            num_tn = np.sum((y_ >= xi) * (y < 0.5))
            s1[i] = (num_fn / total_ture) * 100
            s2[i] = (num_tn / total_false) * 100
        return x, s1, s2

    label, pred_label, pred_score = zip(*data)
    x, s1, s2 = test_thr(label, pred_label, pred_score)

    inds = (s1 < s1_thr) * (s2 > s2_thr)
    x, s1, s2 = x[inds], s1[inds], s2[inds]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    xticklabels = ["{:.2f}".format(xi) for xi in x]
    xticks = np.arange(len(x))
    ax1.plot(s1, "g+")
    ax1.set_ylabel("S1")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    ax2.plot(s2, "r+")
    ax2.set_ylabel("S2")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    if out_file is not None:
        plt.savefig(out_file, dpi=300)
    else:
        plt.show()

    print("\n".join("{:.2f}-{:.2f}-{:.2f}".format(*v) for v in zip(x, s1, s2)))
    return x, s1, s2


def test():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders
    test_loader = loader.construct_test_loader()
    dataset = test_loader.dataset
    # Enable eval mode
    res = []
    model.eval()
    for inputs, labels in test_loader:
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        preds = F.softmax(preds, dim=1)
        # Find the top max_k predictions for each sample
        topk_vals, topk_inds = torch.topk(preds, 1, dim=1)
        # (batch_size, max_k) -> (max_k, batch_size)
        topk_inds, topk_vals = topk_inds.t(), topk_vals.t()
        repk_labels = labels.view(1, -1).expand_as(topk_inds)
        for a, b, c in zip(repk_labels.tolist()[0], topk_inds.tolist()[0], topk_vals.tolist()[0]):
            res.append([a, b, c])

    im_paths = [v["im_path"] for v in dataset.get_imdb()]
    class_ids = ["{}-{}".format(i, v) for i, v in enumerate(dataset.get_class_ids())]

    lines = []
    lines.append(":".join(class_ids))
    lines.append("\nimages,{},res_len,{}\n".format(len(im_paths), len(res)))
    lines.append("im_path,label,pred_label,pred_score")
    for im_path, (a, b, c) in zip(im_paths, res):
        lines.append("{},{},{},{}".format(im_path, a, b, c))

    task_num = time.strftime("%m%d%H%M")

    temp_file = "res_{}.png".format(task_num)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    search_thr(res, s1_thr=3, s2_thr=70, out_file=temp_file)

    temp_file = "res_{}.csv".format(task_num)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    with open(temp_file, "w") as f:
        f.write("\n".join(lines))
        print(temp_file)
    return im_paths, res


def main():
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    test()


if __name__ == "__main__":
    main()
