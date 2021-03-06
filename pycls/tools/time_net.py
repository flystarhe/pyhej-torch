"""Compute model and loader timings."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg


def main():
    config.load_cfg_fom_args("Compute model and loader timings.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)


if __name__ == "__main__":
    main()
