# Data
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

## symlinks
```
ABS_DATA_ROOT = ""
DATA_ROOT = "pygan/datasets/data"

!rm -rf {DATA_ROOT}/abnormal
!ln -s {ABS_DATA_ROOT} {DATA_ROOT}/abnormal
```
