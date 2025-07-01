# A Semi-Supervised ViT Approach

### TLDR

1.  **Initial Training**: A `Vision Transformer (ViT)` is trained on the 682 labeled images using 5-fold cross-validation, achieving an initial score of 0.93
2.  **Pseudo-Labeling**: High-confidence predictions (≥0.96) and core samples from K-Means clustering on the test set are used to generate ~113 high-quality pseudo-labels
3.  **Final Training**: The ViT is retrained on a merged dataset of original and pseudo-labeled data (~795 samples), and a weighted ensemble of all 10 models (5 from Phase 1 + 5 from Phase 3) produces the final 0.97 score


### Solution Overview

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10851731%2Feb19bc16009c1db572751aafcb02dbec%2Foverview.png?generation=1750617600350718&alt=media)

My approach can be broken down into three main phases:

### Phase 1: Initial Supervised Training

This stage aimed to build the strongest possible baseline using only the labeled train data

#### Training Details
*   **Epochs**: `50`
*   **Batch Size**: `8`
*   **Optimizer**: `AdamW` with differential learning rates (backbone: 10% of base LR, head: full base LR)
*   **Learning Rate**: A `CosineAnnealingWarmRestarts` scheduler started with an LR of `1e-4` and decayed it to a minimum of `1e-6`
*   **Weight Decay**: 0.01 for weights, 0.0 for biases and normalization layers
*   **Early Stopping**: To prevent overfitting, training was stopped if the validation score did not improve by at least 0.001 for `5` consecutive epochs

#### Loss Function

**Focal Loss** (`gamma=2.0`) was chosen as the objective function. Focal Loss helps by down-weighting easy, correct predictions, forcing the model to focus its efforts on the more difficult or ambiguous samples, which is exactly what's needed to push the F1 score higher. The loss function also supports confidence weighting, allowing pseudo-labeled samples to be weighted by their prediction confidence during training

I calculated class weights using the **"effective number of samples"** method with β=0.9999. This adjusts weights based on how many examples each class has, so the model doesn't get biased toward the majority class. The method comes from the paper [_Class-Balanced Loss Based on Effective Number of Samples_](https://arxiv.org/pdf/1901.05555), and its official [implementation in PyTorch](https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py#L53-L91). These calculated weights were then passed to the Focal Loss function during training

#### Augmentations

As ViT expects massive amounts of data, and due to the presence of imbalanced classes, I applied different augmentation techniques. These help the model generalize better and avoid overfitting to the limited examples. I used the [`albumentations`](https://albumentations.ai/) library, which includes:

*   **Geometric**: `HorizontalFlip` (p=0.5), `Affine` (scaling: 0.9-1.1, translation: 0-10%, rotation: ±15°, shear: 5°, p=0.7)
*   **Color & Contrast**: `RandomBrightnessContrast` (brightness/contrast: ±0.2, p=0.5), `CLAHE` (clip_limit=2.0, p=0.2), `HueSaturationValue` (hue: ±10, sat: ±15, val: ±10, p=0.3)
*   **Distortion & Blur**: `MotionBlur`/`MedianBlur` (blur_limit=3, p=0.2), `GridDistortion` (distort_limit=0.03, p=0.1), `ElasticTransform` (alpha=1, sigma=50, p=0.1)
*   **Occlusion & Weather**: `CoarseDropout` (3-6 holes, 10-32px, p=0.3), `RandomFog` (fog_coef: 0.1-0.3, p=0.2), `RandomRain` (blur_value=3, p=0.1)
*   **Normalization**: Standard ImageNet mean and std

#### Models Architecture

The core architecture is the **Vision Transformer (ViT)**. I used the [`vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) model, pre-trained on ImageNet-21k. The final classifier head was replaced with a custom one for our 7 classes, featuring a linear layer, batch normalization, GELU activation, and a `Dropout` layer with a probability of 0.4 for regularization. The weights of this custom head were initialized using Xavier Uniform initialization for stable training from the start

The initial training, using 5-fold cross-validation on the original 682 samples, resulted in an ensemble that achieved an F1 score of **0.93** on the leaderboard

### Phase 2: Semi-Supervised Learning & Pseudo-Labeling

This was the key to improving the score. I used two techniques to generate high-quality pseudo-labels from the test set:

1.  **High-Confidence Predictions:** Using the ensemble from Phase 1, I generated predictions on the test set. Any prediction with a confidence score **≥ 0.96** was kept, yielding approximately **~79 pseudo-labeled samples**

2.  **Clustering for Core Samples:** To find more diverse and representative samples, I performed clustering:
    *   Extracted high-dimensional feature embeddings from the penultimate layer of the 5 trained models
    *   Applied UMAP to reduce the dimensionality to 2D, followed by K-Means clustering (k=7)
    *   After verifying that the cluster purity was high (≥ 90%), I extracted the samples closest to each cluster centroid, resulting in about **~34 core samples**

### Phase 3: Final Training & Inference
*   **Merged Dataset**: The original 682 training images were merged with the ~113 pseudo-labeled samples, creating a new, enriched training set of **~795 images**
*   **Final Training**: A new set of 5 ViT models was trained on this merged dataset, using the exact same configuration as in Phase 1
*   **Final Inference**: The final prediction was a **weighted ensemble of all 10 models** (5 from Phase 1 + 5 from Phase 3), giving more weight to the models with higher validation scores

This final ensemble yielded the LB score of **0.97**

### Suggestions

Applying self-supervised learning to this pipeline could further enhance performance. Pre-training the ViT backbone on the test set using a pretext task (e.g., rotation prediction) would allow the model to learn domain-specific features without needing labels. This can make feature extraction and clustering even more effective before pseudo-labeling, especially in low-label regimes. I didn't have time to explore this, but here is a great reference: [Lilian Weng's overview on self-supervised representation learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)

### Conclusion

This semi-supervised approach proved to be very effective in producing high quality synthetic data, demonstrating the power of leveraging unlabeled test data through a combination of high-confidence thresholding and clustering

Thanks for reading, and best of luck to everyone!

### References
[1] [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555)

[2] [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/pdf/2106.10270)

[3] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)

[4] [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/pdf/1802.03426)

[5] [Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)


### Resources

1. [Notebook](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model/notebook)
2. [GitHub repository](https://github.com/ahmedsalim3/sheep-classification-challenge)
