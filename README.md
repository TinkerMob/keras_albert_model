Unofficial implementation of [ALBERT](https://arxiv.org/pdf/1909.11942.pdf).

You can load pretrained model provided by [brightmart/albert_zh](https://github.com/brightmart/albert_zh):

```python
from keras_albert_model import load_brightmart_albert_zh_checkpoint

model = load_brightmart_albert_zh_checkpoint('path_to_checkpoint_folder')
```
