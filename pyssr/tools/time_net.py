"""Compute model and loader timings."""

import pyssr.core.config as config
import pyssr.core.distributed as dist
import pyssr.core.trainer as trainer
from pyssr.core.config import cfg


def main():
    config.load_cfg_fom_args("Compute model and loader timings.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)


if __name__ == "__main__":
    main()
