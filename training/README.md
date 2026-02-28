# MNIST Robust Training (Local, GPU)

This script trains frontend-compatible models and exports JSON weights directly into the app.
It now supports:

- `emnist_digits` from local IDX `.gz` files in `training/` (recommended)
- `local_eval` fallback from `mnist_eval_1000.json` (small)

## Train on GPU 0

```powershell
python training/train_mnist_models.py --gpu-index 0
```

## Faster smoke run

```powershell
python training/train_mnist_models.py --gpu-index 0 --mlp-epochs 8 --cnn-epochs 10
```

## Force dataset source

```powershell
python training/train_mnist_models.py --gpu-index 0 --dataset emnist_digits
python training/train_mnist_models.py --gpu-index 0 --dataset local_eval
```

## Notes

- Exports overwrite:
  - `app/src/data/deep-learning/models/mnist/mlp/model.json`
  - `app/src/data/deep-learning/models/mnist/mlp/weights-shard1.json`
  - `app/src/data/deep-learning/models/mnist/cnn/model.json`
  - `app/src/data/deep-learning/models/mnist/cnn/weights-shard1.json`
- Models stay compatible with the existing visualizer:
  - MLP: single hidden layer (`w1/b1`, `w2/b2`)
  - CNN: single 3x3 conv + maxpool + dense (`kernels`, `dense_w`, `dense_b`)
