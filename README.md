# Sheep Classification Challenge

A computer vision project to classify Arabian sheep breeds from images using machine learning. Built for a cultural challenge celebrating Eid al-Adha.
Implemnting a state-of-the-art hybrid image classifier combining Vision Transformer (ViT) and CLIP models. The fusion of these architectures leverages both local feature extraction (ViT) and semantic understanding (CLIP), read more [here](./docs/method.md)

## Key Features
- Dual-model architecture (ViT + CLIP)
- Multiple fusion methods (concatenation, attention)

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

Edit YAML files in `configs/`:

    - [paths.yml](./configs/paths.yml) - Dataset paths and output directories
    - [model.yml](./configs/model.yml) - Model architecture and training parameters

## Dataset

The dataset can be found on [Kaggle](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/data).
You can also download it by running `make download-data`. Make sure to configure your Kaggle credential paths [here](./scripts/download_dataset.sh#L11)

## Training

```sh
make train
# or python scripts/train_vit_clip.py
```
After training, predictions will be saved in CSV format at: [output/submission.csv](./output/submission.csv)

## Team Structure and Contribution

[@ahmedsalim3](https://github.com/ahmedsalim3)

## References

[Eid Al-Adha 2025: Sheep Classification Challenge](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview)
