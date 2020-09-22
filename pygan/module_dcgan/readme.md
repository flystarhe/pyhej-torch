# DCGAN
[GitHub](https://github.com/pytorch/examples/tree/master/dcgan)

## Data
Abnormal:
```
abnormal
├── train
│   ├── abnormal_large
│   │   ├── 1.png
│   │   └── 1.xml
│   ├── abnormal_small
│   │   ├── 2.png
│   │   └── 2.xml
│   ├── contamination
│   │   ├── 3.png
│   │   └── 3.xml
│   └── ok
│       ├── 4.png
│       ├── 4.xml
│       ├── 5.png
│       └── 5.xml
└── val
    ├── abnormal_large
    │   ├── 11.png
    │   └── 11.xml
    ├── abnormal_small
    │   ├── 12.png
    │   └── 12.xml
    ├── contamination
    │   ├── 13.png
    │   └── 13.xml
    └── ok
        ├── 14.png
        └── 14.xml
```

## SYRB
```
%matplotlib inline
import os
PYHEJ_TORCH = "/data/sdv1/tmps/gits/pyhej-torch"
os.environ["PYHEJ_TORCH"] = PYHEJ_TORCH
os.chdir(PYHEJ_TORCH)
!git log -1

ABS_DATA_ROOT = "/data/sdv1/tmps/syrb/test_0909_cvt"
DATA_ROOT = "pygan/datasets/data"
!rm -rf {DATA_ROOT}/abnormal
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/abnormal

ARG_CFG = "pygan/module_dcgan/syrb_dcgan_4gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/syrb/test_0909_a1"
ARGS = "--cfg {} OUT_DIR {} RNG_SEED 1 LOG_DEST stdout LOG_PERIOD 100 TRAIN.AUTO_RESUME False".format(ARG_CFG, ARG_OUT_DIR)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` python pygan/module_dcgan/py_train.py {ARGS}

ARG_CFG = "pygan/module_dcgan/syrb_dcgan_4gpu.yaml"
ARG_OUT_DIR = "/data/sdv1/tmps/syrb/test_0909_a1"
ARG_WEIGHTS = "{}/checkpoints/model_epoch_0100.pyth".format(ARG_OUT_DIR)
ARGS = "--cfg {} OUT_DIR {} TEST.WEIGHTS {} RNG_SEED 1 NUM_GPUS 1".format(ARG_CFG, ARG_OUT_DIR, ARG_WEIGHTS)
!PYTHONPATH={PYHEJ_TORCH}:`pwd` python pygan/module_dcgan/py_app.py {ARGS}
```

## 参考资料：
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [DCGAN ON FASHIONGEN](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_dcgan/)
