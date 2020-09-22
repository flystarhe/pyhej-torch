import pygan.core.config as config
import pygan.core.distributed as dist
import pygan.core.trainer as trainer
from pygan.core.config import cfg


def main():
    config.load_cfg_fom_args("Test a trained model.")
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model)


if __name__ == "__main__":
    main()
