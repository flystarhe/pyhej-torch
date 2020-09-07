# ReadMe
* PyTorch 1.3+
* tested with CUDA 9.2 and cuDNN 7.1
* github.com/facebookresearch/pycls (-54-c923a4d)

```
$ pip install -r requirements.txt
```

## jupyter
```
%matplotlib inline
import os
PYHEJ_TORCH = "/mnt/d/work/gits/pyhej-torch"
os.environ["PYHEJ_TORCH"] = PYHEJ_TORCH
os.chdir(PYHEJ_TORCH)
!git log -1

ABS_DATA_ROOT = ""
DATA_ROOT = "pycls/datasets/data"
!rm -rf {DATA_ROOT}/imagenet
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/imagenet

!PYTHONPATH={PYHEJ_TORCH}:`pwd` nohup python pycls/tools/train_net.py *args >> tmp/log.00 2>&1 &
```

## data
Custom:
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

ImageNet:
```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

CIFAR-10:
```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

## ref
* https://github.com/facebookresearch/pycls
* https://github.com/facebookresearch/fvcore
