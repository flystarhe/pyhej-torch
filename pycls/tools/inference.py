import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
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


def search_thr(outputs, class_ids, min_p=80, min_r=50, out_file=None):
    """Search from: [(im_path,label,pred_label,pred_score),]"""
    def test_thr(tag, label, pred_label, pred_score):
        y = np.array([tag == l for l in label])
        y_ = np.array([tag == l for l in pred_label])
        score_ = np.array(pred_score, dtype="float32")

        x = np.linspace(0, 1, 100, endpoint=False)[1:]
        p = np.zeros_like(x)
        r = np.zeros_like(x)
        n_gt = np.sum(y)
        for i, xi in enumerate(x):
            n_dt = np.sum(y_ * (score_ >= xi))
            n_tp = np.sum(y * y_ * (score_ >= xi))
            p[i] = n_tp / n_dt * 100 if n_tp > 0 else 0
            r[i] = n_tp / n_gt * 100 if n_tp > 0 else 0
        return x, p, r

    _, label, pred_label, pred_score = zip(*outputs)

    n_class = len(class_ids)
    _, axs = plt.subplots(2 * n_class, figsize=(8 * n_class, 8))

    score_thr = {}
    for i, class_id in enumerate(class_ids):
        x, p, r = test_thr(class_id, label, pred_label, pred_score)

        inds = (p >= min_p) * (r >= min_r)
        if np.any(inds):
            x, p, r = x[inds], p[inds], r[inds]

        best_id = ((p - min_p) / min_p + (r - min_r) / min_r).argmax()
        score_thr[class_id] = x[best_id]

        xticklabels = ["{:.2f}".format(xi) for xi in x]
        xticks = np.arange(len(x))

        viz_step = ((xticks.size - 1) // 20 + 1)
        xticklabels = xticklabels[::viz_step]
        xticks = xticks[::viz_step]

        axs[2 * i + 0].plot(p, "g+")
        axs[2 * i + 0].set_ylabel("P")
        axs[2 * i + 0].set_xticks(xticks)
        axs[2 * i + 0].set_xticklabels(xticklabels)
        axs[2 * i + 0].set_title("{:.2f}, {:.2f}".format(x[best_id], p[best_id]))
        axs[2 * i + 1].plot(r, "r+")
        axs[2 * i + 1].set_ylabel("R")
        axs[2 * i + 1].set_xticks(xticks)
        axs[2 * i + 1].set_xticklabels(xticklabels)
        axs[2 * i + 1].set_title("{:.2f}, {:.2f}".format(x[best_id], r[best_id]))

    if out_file is not None:
        plt.savefig(out_file, dpi=300)
    else:
        plt.show()

    print("Threshold:", json.dumps(score_thr, indent=4, sort_keys=True))
    return score_thr


def hardmini(outputs, class_ids, task_name, score_thr=None):
    out_dir = "{}/hardmini".format(task_name)
    out_dir = os.path.join(cfg.OUT_DIR, out_dir)

    if score_thr is None:
        score_thr = dict()

    for ori_label in class_ids:
        for out_label in class_ids:
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_U")
            os.makedirs(sub_dir, exist_ok=True)
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_D")
            os.makedirs(sub_dir, exist_ok=True)

    for im_path, ori_label, out_label, out_score in outputs:
        if out_score >= score_thr.get(out_label, 0.5):
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_U")
        else:
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_D")
        shutil.copyfile(im_path, os.path.join(sub_dir, os.path.basename(im_path)))
    return out_dir


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
    logs = []
    model.eval()
    for inputs, labels in test_loader:
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        if cfg.USE_SIGMOID:
            preds = torch.sigmoid(preds)
        else:
            preds = F.softmax(preds, dim=1)
        # Abnormal dataset format support
        if cfg.TEST.DATASET == "abnormal":
            col = torch.ones(labels.size(0), 1, dtype=labels.dtype)
            labels = torch.cat((col, labels * 2), dim=1)
            labels = labels.argmax(dim=1)

            col = 1 - preds.max(dim=1, keepdim=True)[0]
            preds = torch.cat((col, preds), dim=1)
        # Find the top max_k predictions for each sample
        topk_vals, topk_inds = torch.topk(preds, 5, dim=1)
        # (batch_size, max_k) -> (max_k, batch_size)
        topk_inds, topk_vals = topk_inds.t(), topk_vals.t()
        repk_labels = labels.view(1, -1).expand_as(topk_inds)
        for a, b, c in zip(repk_labels.tolist()[0], topk_inds.tolist()[0], topk_vals.tolist()[0]):
            logs.append([a, b, c])

    imgs = [v["im_path"] for v in dataset._imdb]
    class_ids = dataset._class_ids
    assert len(imgs) == len(logs)

    if cfg.TEST.DATASET == "abnormal":
        class_ids = ["ok"] + class_ids

    lines = []
    outputs = []
    lines.append(":".join(class_ids))
    lines.append("\nimages,{}\n".format(len(imgs)))
    lines.append("im_path,label,pred_label,pred_score")

    for im_path, (a, b, c) in zip(imgs, logs):
        lines.append("{},{},{},{}".format(im_path, a, b, c))
        outputs.append([im_path, class_ids[a], class_ids[b], c])

    task_name = time.strftime("%m%d%H%M")
    os.makedirs(os.path.join(cfg.OUT_DIR, task_name))

    temp_file = "{}/threshold.png".format(task_name)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    score_thr = search_thr(outputs, class_ids, min_p=80, min_r=50, out_file=temp_file)

    temp_file = "{}/results.csv".format(task_name)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    with open(temp_file, "w") as f:
        f.write("\n".join(lines))
        print(temp_file)

    temp_file = "{}/results.pkl".format(task_name)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    with open(temp_file, "wb") as f:
        pickle.dump(outputs, f)
        print(temp_file)

    hardmini(outputs, class_ids, task_name, score_thr)
    return outputs


def main():
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    test()


if __name__ == "__main__":
    main()
