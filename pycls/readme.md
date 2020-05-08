# ReadMe
* PyTorch 1.3
* tested with CUDA 9.2 and cuDNN 7.1

```
$ pip install -r requirements.txt
```

## sh
```
$ PYCLS=/path/to/pyhej-torch
$ PYTHONPATH=$PYCLS nohup python tools/net_train.py *args > tmp/log.00 2>&1 &
```

## data
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

Symlink:
```
$ mkdir -p /path/pycls/datasets/data
$ ln -s /path/imagenet /path/pycls/datasets/data/imagenet
$ ln -s /path/cifar10 /path/pycls/datasets/data/cifar10
```

>注意，删除软连接时末尾没有斜杠`rm -rf /path/pycls/datasets/data`

## ref
* https://github.com/facebookresearch/pycls
* https://github.com/facebookresearch/fvcore
