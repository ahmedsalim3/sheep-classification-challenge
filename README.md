# Sheep Classification Challenge

A deep learning solution for classifying sheep breeds using semi-supervised learning. The project tackles a small, imbalanced dataset with smart data mining techniques and Vision Transformers.

👉 Check the [project page](https://ahmedsalim3.github.io/sheep-classification-challenge/), it walks through everything: training strategy, tricks used, and what worked (and what didn't)


## How to install

1. Clone this reop

```bash
git clone git@github.com:ahmedsalim3/sheep-classification-challenge.git
cd sheep-classification-challenge
```

2. Install dependencies

```sh
make install
```

## Configuration

Edit YAML file in [`config.yml`](./configs/config.yml) to configure the paths, output directories, and training parameters

## Dataset

The dataset is hosted on [Kaggle](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/data).

To download it automatically:

```sh
make download-data
```

Make sure your Kaggle credentials are set correctly [here](./scripts/download_dataset.sh#L11)

## Training

The project supports two training modes:

### Cross-Validation Training

For robust model evaluation with k-fold cross-validation:

```sh
python scripts/train.py --use_cross_validation
```

### Normal Training

For faster training with train/validation split:

```sh
python scripts/train.py --val_split 0.2
```

### Custom Validation Split

```sh
python scripts/train.py --val_split 0.3
```

It supports clean training, pseudo-labeling, clustering, and full semi-supervised loops

## Results

All final logs, metrics, and plots can be found inside the [results](./results/) folder

## References

[Eid Al-Adha 2025: Sheep Classification Challenge](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview)

[Fix The Data First, Then Worry About The Model](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model/notebook)

## Repo Structure

```sh
sheep-classification-challenge/
├── src/
│   ├── modeling/         # Training, evaluation, clustering
│   ├── data/             # Dataset, transforms, pseudo-labeling
│   └── utils/            # Helpers and utilities
├── scripts/
│   ├── train.py          # Unified training script
│   ├── train_cv.sh       # Cross-validation training
│   ├── train_normal.sh   # Normal training
│   └── download_dataset.sh
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks
└── results/              # Training outputs and metrics
```
