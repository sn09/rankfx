# rankfx

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

## Minimal working example

Download package to your environment:

```bash
pip install -U rankfx
```

Load data and start training and evaluating processes:
```python
import pandas as pd
from rankfx.dcnv2.model import DCNv2


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_val = pd.read_csv("valid.csv")

dcnv2_model = DCNv2(
    model_structure="stacked_parallel",
    use_low_rank_mixture=True,
    cross_low_rank_dim=32,
    num_cross_layers=5,
    num_cross_experts=4,
    parallel_hidden_dims=[400, 400, 400],
    parallel_dropout=0.2,
    parallel_use_batch_norm=True,
    parallel_activation=nn.ReLU,
    stacked_hidden_dims=[500, 500, 500],
    stacked_dropout=0.2,
    stacked_use_batch_norm=True,
    stacked_activation=nn.ReLU,
    output_dim=1,
    proj_output_embeddings=False,
)

train_metrics_dcnv2, val_metrics_dcnv2 = dcnv2_model.fit(
    features=df_train.drop(columns="label"),
    target=df_train["label"],
    val_features=df_val.drop(columns="label"),
    val_target=df_val["label"],
    optimizer_cls="torch.optim.Adam",
    optimizer_params=dict(lr=1e-3),
    scheduler_cls="torch.optim.lr_scheduler.CosineAnnealingLR",
    scheduler_params=dict(T_max=20),
    num_epochs=20,
    seed=42,
    artifacts_path="./dcnv2_artifacts",
    device="cuda:0",
    batch_size=4096,
    num_workers=2,
    eval_metric_name="log_loss",
    eval_mode="min",
    embedded_features=["user_id", "item_id", "tag_id"],
    default_embedding_size=64,
    oov_masking_proba=0.05,
)
test_metrics_dcnv2 = dcnv2_model.test(
    features=df_test.drop(columns="label"),
    target=df_test["label"],
    device="cuda:0",
    batch_size=4096,
    num_workers=2,
)
```

## Interface

To wrap your own model into provided interface, you should do following steps:

1. Inherit your PyTorch model from `models/common/base/model/NNPandasModel`
2. Implement following methods:  
  a. `forward` - model forward pass  
  b. `train_step` - train stage step, should take batch of data and return train metrics as dict, MUST have `loss` key in output  
  c. `val_step/test_step` - validation and test stage steps, should take batch of data and return metrics as dict  
  d. `inference_step` - inference stage step, should take batch of data and return model output  
  e. `_init_modules` - instantiate torch modules based on `models/common/features/config/FeaturesConfig`, which will be infered from train data  
