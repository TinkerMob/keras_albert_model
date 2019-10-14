# ALBERT

Unofficial implementation of [ALBERT](https://arxiv.org/pdf/1909.11942.pdf).

## Install

```bash
python setup.py install
```

or

```bash
pip install git+https://github.com/TinkerMob/keras_albert_model.git
```

Current versions of dependencies:

* keras==2.3.0
* tensorflow==2.0.0

## Build model

```python
from keras_albert_model import build_albert

model = build_albert(token_num=30000, training=True)
model.summary()
```

## Load checkpoint

You can load pretrained model provided by [brightmart/albert_zh](https://github.com/brightmart/albert_zh):

```python
from keras_albert_model import load_brightmart_albert_zh_checkpoint

model = load_brightmart_albert_zh_checkpoint('path_to_checkpoint_folder')
model.summary()
```

## Select output layers

```python
from keras_albert_model import build_albert

model = build_albert(token_num=30000, training=False, output_layers=[-1, -2, -3, -4])
model.summary()
```
