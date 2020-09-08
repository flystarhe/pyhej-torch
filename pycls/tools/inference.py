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


def onehot(pos, size):
    return [1 if i == pos else 0 for i in range(size)]


def search_thr(outputs, class_ids, min_p=80, min_r=50, out_file=None):
    """Search from: [(im_path,ori_label,out_label,out_score,),]"""
    def test_thr(label, score):
        y = np.array(label, dtype="float32")
        y_ = np.array(score, dtype="float32")

        x = np.linspace(0, 1, 100, endpoint=False)[1:]
        p, r = np.zeros_like(x), np.zeros_like(x)
        for i, xi in enumerate(x):
            n_gt = np.sum((y >= 0.5))
            n_dt = np.sum((y_ >= xi))
            n_tp = np.sum((y >= 0.5) * (y_ >= xi))
            p[i] = n_tp / n_dt * 100 if n_tp > 0 else 0
            r[i] = n_tp / n_gt * 100 if n_tp > 0 else 0
        return x, p, r

    _, _, _, _, _label_rows, _score_rows = zip(*outputs)

    n = len(class_ids)
    if isinstance(_label_rows[0], int):
        _label_rows = [onehot(v, n) for v in _label_rows]

    _, axs = plt.subplots(2 * n, figsize=(6, 6 * n))

    score_thr = {}
    _label_cols = np.array(_label_rows, dtype="float32").T
    _score_cols = np.array(_score_rows, dtype="float32").T
    assert _label_cols.shape == _score_cols.shape, "label same shape as pred"
    for i, (class_id, _label_col, _score_col) in enumerate(zip(class_ids, _label_cols, _score_cols)):
        x, p, r = test_thr(_label_col, _score_col)

        inds = (p >= min_p) * (r >= min_r)
        if np.any(inds):
            x, p, r = x[inds], p[inds], r[inds]

        best_id = ((p - min_p) / min_p + (r - min_r) / min_r).argmax()
        score_thr[class_id] = x[best_id]
        score_thr[class_id + "_tail"] = "P:{:.3f},R:{:.3f}".format(p[best_id], r[best_id])

        xticklabels = ["{:.2f}".format(xi) for xi in x]
        xticks = np.arange(len(x))

        viz_step = ((xticks.size - 1) // 20 + 1)
        xticklabels = xticklabels[::viz_step]
        xticks = xticks[::viz_step]

        axs[2 * i + 0].plot(p, "g+")
        axs[2 * i + 0].set_xticks(xticks)
        axs[2 * i + 0].set_xticklabels(xticklabels, fontdict={"fontsize": "xx-small"})
        axs[2 * i + 0].set_ylabel("P({})".format(class_id), fontdict={"fontsize": "small"})
        axs[2 * i + 0].set_title("P: {:.2f}, {:.2f}".format(x[best_id], p[best_id]), fontdict={"fontsize": "small"})
        axs[2 * i + 1].plot(r, "r+")
        axs[2 * i + 1].set_xticks(xticks)
        axs[2 * i + 1].set_xticklabels(xticklabels, fontdict={"fontsize": "xx-small"})
        axs[2 * i + 1].set_ylabel("R({})".format(class_id), fontdict={"fontsize": "small"})
        axs[2 * i + 1].set_title("R: {:.2f}, {:.2f}".format(x[best_id], r[best_id]), fontdict={"fontsize": "small"})

    plt.show()
    if out_file is not None:
        plt.savefig(out_file, dpi=300)

    print("Threshold:", json.dumps(score_thr, indent=4, sort_keys=True))
    return score_thr


def hardmini(outputs, class_ids, task_name, score_thr=None):
    out_dir = "{}_hardmini".format(task_name)
    out_dir = os.path.join(cfg.OUT_DIR, out_dir)

    if score_thr is None:
        score_thr = dict()

    for ori_label in class_ids:
        for out_label in class_ids:
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_U")
            os.makedirs(sub_dir, exist_ok=True)
            sub_dir = os.path.join(out_dir, ori_label + out_label + "_D")
            os.makedirs(sub_dir, exist_ok=True)

    for im_path, ori_label, out_label, out_score, _, _ in outputs:
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
        if cfg.SOFTMAX:
            preds = F.softmax(preds, dim=1)
        else:
            preds = torch.sigmoid(preds)
        # Abnormal dataset format support
        repk_labels = labels
        if cfg.TRAIN.DATASET == "abnormal":
            repk_labels = labels.argmax(dim=1)
        repk_labels = repk_labels.view(1, -1).tolist()
        # Find the top max_k predictions for each sample
        topk_vals, topk_inds = torch.topk(preds, 1, dim=1)
        # (batch_size, max_k) -> (max_k, batch_size)
        topk_vals, topk_inds = topk_vals.t().tolist(), topk_inds.t().tolist()
        for a, b, c, _label, _score in zip(repk_labels[0], topk_inds[0], topk_vals[0], labels.tolist(), preds.tolist()):
            logs.append([a, b, c, _label, _score])

    imgs = [v["im_path"] for v in dataset._imdb]
    class_ids = dataset._class_ids
    assert len(imgs) == len(logs)

    lines = []
    outputs = []
    lines.append(":".join(class_ids))
    lines.append("images,{}".format(len(imgs)))
    lines.append("im_path,ori_label,out_label,out_score,out_score_1_n]")

    for im_path, (a, b, c, _label, _score) in zip(imgs, logs):
        _score_str = ",".join(["{:.3f}".format(v) for v in _score])
        lines.append("{},{},{},{},{}".format(im_path, a, b, c, _score_str))
        outputs.append([im_path, class_ids[a], class_ids[b], c, _label, _score])

    task_name = time.strftime("%m%d%H%M%S")
    os.makedirs(os.path.join(cfg.OUT_DIR, task_name))

    temp_file = "{}/threshold.png".format(task_name)
    temp_file = os.path.join(cfg.OUT_DIR, temp_file)
    score_thr = search_thr(outputs, class_ids, min_p=70, min_r=98, out_file=temp_file)

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
