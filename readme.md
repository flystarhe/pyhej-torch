# Pyhej Torch
Deep Learning Grocery Store.(Python 3.8+)

## Jupyter
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

EXT_PATH = ""
EXT_ARGS = ""
!PYTHONPATH={EXT_PATH}:`pwd` nohup python pycls/tools/train_net.py {EXT_ARGS} >> tmp/log.00 2>&1 &
```

## Sphinx
* `cd docs`进入文档目录
* `sphinx-quickstart`初始化
* 在`docs/source/conf.py`中完善配置
* 生成API文档`sphinx-apidoc -o source ../`
* 生成文档`make html`
* 清理`make clean`

```
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

...

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
```
