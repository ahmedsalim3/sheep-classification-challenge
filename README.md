# Sheep Classification Challenge

A deep learning solution for classifying sheep breeds with semi-supervised learning.


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

The dataset can be found on [Kaggle](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/data).
You can also download it by running `make download-data`. Make sure to configure your Kaggle credential paths [here](./scripts/download_dataset.sh#L11)

## Training

```sh
make train
# or python scripts/train.py
```

## Results

The complete results can be found [here](./results/)

## Team Structure and Contribution

[@ahmedsalim3](https://github.com/ahmedsalim3)

## References

[Eid Al-Adha 2025: Sheep Classification Challenge](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview)
[Fix The Data First, Then Worry About The Model](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model/notebook)
