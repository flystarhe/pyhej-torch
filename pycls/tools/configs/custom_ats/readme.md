# ATS

## Data
```
custom
|_ train
|  |_ c1
|  |_ ...
|  |_ cn
|_ val
|  |_ c1
|  |_ ...
|  |_ cn
|_ ...
```

Symlink:
```
ABS_DATA_ROOT = ""
DATA_ROOT = "pycls/datasets/data"

!rm -rf {DATA_ROOT}/custom
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/custom
```

## Note
```
%matplotlib inline
import os
PYHEJ_TORCH = "/data/sdv1/tmps/gits/pyhej-torch"
os.environ["PYHEJ_TORCH"] = PYHEJ_TORCH
os.chdir(PYHEJ_TORCH)
!git log -1

ABS_DATA_ROOT = "/data/sdv1/tmps/ats/task_0902_padded"
DATA_ROOT = "pycls/datasets/data"
!rm -rf {DATA_ROOT}/custom
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/custom

ARG_CFG = "pycls/tools/configs/custom_ats/R-50-1x64d_step_8gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/ats/task_0902_padded_a1"
ARGS = "--cfg {} OUT_DIR {} RNG_SEED 1 LOG_DEST stdout LOG_PERIOD 500 TRAIN.AUTO_RESUME False".format(ARG_CFG, ARG_OUT_DIR)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` python pycls/tools/train_net.py {ARGS}

ARG_CFG = "pycls/tools/configs/custom_ats/R-50-1x64d_step_8gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/ats/task_0902_padded_a1"
ARG_WEIGHTS = "{}/checkpoints/model_epoch_0100.pyth".format(ARG_OUT_DIR)
ARGS = "--cfg {} OUT_DIR {} TEST.WEIGHTS {} RNG_SEED 1 NUM_GPUS 1".format(ARG_CFG, ARG_OUT_DIR, ARG_WEIGHTS)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` python pycls/tools/ats_inference.py {ARGS}
```
