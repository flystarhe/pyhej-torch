from collections import deque

import numpy as np
import pyssr.core.logging as logging
import torch
from pyssr.core.config import cfg
from pyssr.core.timer import Timer


logger = logging.get_logger(__name__)


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def compute_error(preds, targets):
    """Computes the error."""
    err_str = "Size of predictions and targets must match"
    assert preds.size() == targets.size(), err_str
    diff = torch.abs(preds - targets)
    return diff.mean(), diff.var()


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.err_mean = ScalarMeter(cfg.LOG_PERIOD)
        self.err_var = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.err_mean_total = 0.0
        self.err_var_total = 0.0
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.err_mean.reset()
        self.err_var.reset()
        self.err_mean_total = 0.0
        self.err_var_total = 0.0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, err_mean, err_var, loss, lr, mb_size):
        # Current minibatch stats
        self.err_mean.add_value(err_mean)
        self.err_var.add_value(err_var)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.err_mean_total += err_mean * mb_size
        self.err_var_total += err_var * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "err_mean": self.err_mean.get_win_median(),
            "err_var": self.err_var.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        logger.info(logging.dump_log_data(stats, "train_iter"))

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        err_mean = self.err_mean_total / self.num_samples
        err_var = self.err_var_total / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "err_mean": err_mean,
            "err_var": err_var,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, "train_epoch"))


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.err_mean = ScalarMeter(cfg.LOG_PERIOD)
        self.err_var = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full test set)
        self.min_err_mean = np.inf
        self.min_err_var = np.inf
        # Number of misclassified examples
        self.err_mean_total = 0.0
        self.err_var_total = 0.0
        self.num_samples = 0

    def reset(self, min_errs=False):
        if min_errs:
            self.min_err_mean = np.inf
            self.min_err_var = np.inf
        self.iter_timer.reset()
        self.err_mean.reset()
        self.err_var.reset()
        self.err_mean_total = 0.0
        self.err_var_total = 0.0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, err_mean, err_var, mb_size):
        self.err_mean.add_value(err_mean)
        self.err_var.add_value(err_var)
        self.err_mean_total += err_mean * mb_size
        self.err_var_total += err_var * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "err_mean": self.err_mean.get_win_median(),
            "err_var": self.err_var.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        logger.info(logging.dump_log_data(stats, "test_iter"))

    def get_epoch_stats(self, cur_epoch):
        err_mean = self.err_mean_total / self.num_samples
        err_var = self.err_var_total / self.num_samples
        self.min_err_mean = min(self.min_err_mean, err_mean)
        self.min_err_var = min(self.min_err_var, err_var)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "err_mean": err_mean,
            "err_var": err_var,
            "min_err_mean": self.min_err_mean,
            "min_err_var": self.min_err_var,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, "test_epoch"))
