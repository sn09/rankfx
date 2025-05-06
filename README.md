# ranking

This repo provides framework for using models created with PyTorch with Scikit-learn API.  
Main focus is on ranking models, implementations were partially taken from [FuxiCTR](https://github.com/reczoo/FuxiCTR)  
Right now following models are implemented:

- [DCNv2](https://arxiv.org/abs/2008.13535)
- [FinalNet](https://dl.acm.org/doi/10.1145/3539618.3591988)

## Benchmarks

Datasets:
- [Movielens_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1)
- [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1)
- [KKBox_x1](https://github.com/reczoo/Datasets/tree/main/KKBox/KKBox_x1)
- [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1)

All quality measurements are done without any tuning and with not much training epochs.

| No  | Model          | AUC Movielens_x1 | AUC Frappe_x1 | AUC KKBox_x1 | AUC Avazu_x1 |
|:---:|:--------------:|:----------------:|:-------------:|:------------:|:------------:|
|0    | [LGBMClassisifer](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) | 0.93878 | 0.98406 | 0.77265 | 0.75589 |
|1    | [DCNv2](https://arxiv.org/abs/2008.13535) | 0.93801 | 0.96225 | 0.78645 | 0.75401 |
|2    | [FinalNet](https://dl.acm.org/doi/10.1145/3539618.3591988) | 0.94116 | 0.97601 | 0.80282 | 0.75844 |

## Interface

To wrap your own model into provided interface, you should do following steps:

1. Inherit your PyTorch model from `models/common/base/model/NNPandasModel`
2. Implement following methods:  
  a. `forward` - model forward pass  
  b. `train_step` - train stage step, should take batch of data and return train metrics as dict, MUST have `loss` key in output  
  c. `val_step/test_step` - validation and test stage steps, should take batch of data and return metrics as dict  
  d. `inference_step` - inference stage step, should take batch of data and return model output  
  e. `_init_modules` - instantiate torch modules based on `models/common/features/config/FeaturesConfig`, which will be infered from train data  

A bit more about `FeaturesConfig`:
- Will be infered from train data during training
- Contains list of features, each of them has following attributes: description parameters (`name`, `feature_type`, `feature_size` - can be > 1 for sequential features) and embedding_parameters (`needs_embed`, `embedding_size`, `embedding_vocab_size`, `embedding_padding_idx`)

