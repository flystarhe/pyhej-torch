# Data
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

## symlinks
```
ABS_DATA_ROOT = ""
DATA_ROOT = "pycls/datasets/data"

!rm -rf {DATA_ROOT}/imagenet
!ln -s {REL_DATA_ROOT} {DATA_ROOT}/imagenet
```
