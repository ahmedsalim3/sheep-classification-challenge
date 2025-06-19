# Sheep Classification Challenge



## Overview


This project was built for the [Kaggle Sheep Classification Challenge 2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview), where the goal was to classify 7 sheep breeds using just 682 labeled images. With a small, highly imbalanced dataset, a visually noisy image set, and F1-score as the evaluation metric, this task was far from easy.


The core question became:

_How do we make the most of limited data without overfitting or biasing the model?_

## Approach

The project tackled this challenge with a semi-supervised pipeline. We started with a Vision Transformer (ViT) model and built everything around improving data quality:


1. Initial training using 5-fold cross-validation on clean labels
2. Pseudo-labeling the test set using the trained ensemble with a strict confidence threshold (≥ 0.97)
3. KMeans clustering on ViT feature embeddings to mine consistent patterns in the unlabeled data
4. Core sample filtering: we only kept samples close to cluster centroids (purity ≥ 0.9)
5. Final training on a richer dataset: clean + pseudo + clustered

## Results

- Extracted 34 extra samples from clustering, all auto-labeled using feature space similarity

- Ended up with ~115 training samples total after synthetic expansion, using ~78% of the unlabeled set
- Ended up with ~115 training samples total after synthetic expansion, using ~78% from the unlabeled set

- Best score on Kaggle: 0.97046 F1

- Used Focal Loss + Class Weights to handle imbalances gracefully.

- Created a custom training loop, dynamic LR scheduler, and a fair weighting system between clean and pseudo-labeled samples.

## Findings

- Pseudo-labels can be powerful if you set a strict threshold and don't blindly trust them

- Synthetic labels aren’t real labels, but with filtering and consistency, they can get pretty close

- Simple clustering with a purity check can turn noisy unlabeled data into meaningful training signals

## Try it Yourself

All code is open-source, either in the [repo](https://github.com/ahmedsalim3/sheep-classification-challenge) or in the [Kaggle notebook](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model#5.2-K-Mean-Clustering)
