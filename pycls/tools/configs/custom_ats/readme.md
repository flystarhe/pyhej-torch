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
import os

PYHEJ_TORCH = "/data/sdv1/tmps/gits/pyhej-torch"

os.environ["PYHEJ_TORCH"] = PYHEJ_TORCH
os.chdir(PYHEJ_TORCH)
!pwd

ABS_DATA_ROOT = "/data/sdv1/tmps/ats/padded_split_123"
DATA_ROOT = "pycls/datasets/data"

!rm -rf {DATA_ROOT}/custom
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/custom

ARG_CFG = "pycls/tools/configs/custom_ats/R-50-1x64d_step_8gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/results/pycls-resnet50-t1"
ARGS = "--cfg {} OUT_DIR {} RNG_SEED 1 LOG_DEST file LOG_PERIOD 50".format(ARG_CFG, ARG_OUT_DIR)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` nohup python pycls/tools/train_net.py {ARGS} >> tmp/log.00 2>&1 &

ARG_CFG = "pycls/tools/configs/custom_ats/R-50-1x64d_step_8gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/results/pycls-resnet50-t1"
ARG_WEIGHTS = ""
ARGS = "--cfg {} OUT_DIR {} TEST.WEIGHTS {} RNG_SEED 1 NUM_GPUS 1".format(ARG_CFG, ARG_OUT_DIR, ARG_WEIGHTS)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` nohup python pycls/tools/inference.py {ARGS} >> tmp/log.00 2>&1 &
```
