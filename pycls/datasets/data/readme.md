# Data
Abnormal:
```
abnormal
├── train
│   ├── abnormal_large
│   │   ├── 1.png
│   │   └── 2.png
│   ├── abnormal_small
│   │   ├── 3.png
│   │   └── 4.png
│   ├── contamination
│   │   ├── 5.png
│   │   └── 6.png
│   └── ok
│       ├── 7.png
│       ├── 8.png
│       └── 9.png
└── val
    ├── abnormal_large
    │   ├── 10.png
    │   └── 11.png
    ├── abnormal_small
    │   ├── 12.png
    │   └── 13.png
    ├── contamination
    │   ├── 14.png
    │   └── 15.png
    └── ok
        ├── 16.png
        └── 17.png
```

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

## symlinks
```
ABS_DATA_ROOT = ""
DATA_ROOT = "pycls/datasets/data"

!rm -rf {DATA_ROOT}/imagenet
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/imagenet
```
