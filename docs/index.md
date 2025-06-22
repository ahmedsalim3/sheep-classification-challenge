# Sheep Classification Challenge

A deep learning solution for the [Kaggle Sheep Classification Challenge 2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview), achieving **0.97 F1-score** using semi-supervised learning techniques on a small, imbalanced dataset.

## Challenge Overview

The goal was to classify 7 sheep breeds using just **682 labeled images** with significant class imbalance and visually noisy data. The evaluation metric was F1-score, making this a particularly challenging task.

**Key Challenges:**
- Extremely small dataset (682 images)
- High class imbalance across 7 breeds
- Visually noisy images with poor quality
- F1-score evaluation requiring balanced precision/recall

## Solution Approach

Our solution employs a **semi-supervised learning pipeline** built around Vision Transformers (ViT) with intelligent data mining techniques:

### 1. Initial Training
- **5-fold cross-validation** on clean labeled data
- Vision Transformer (ViT) architecture with differential learning rates
- Focal Loss + Effective Class Weights (β=0.9999) for imbalance handling
- CosineAnnealingWarmRestarts scheduler with early stopping

### 2. Pseudo-labeling
- Ensemble predictions on unlabeled test set (144 images)
- **Strict confidence threshold (≥ 0.96)** for quality control
- Extracted ~79 high-confidence pseudo-labeled samples

### 3. Clustering-based Data Mining
- **K-Means clustering** on ViT feature embeddings with UMAP dimensionality reduction
- **Purity threshold (≥ 0.9)** for cluster filtering
- Extracted ~34 high-quality core samples from unlabeled data
- Feature space similarity for automatic labeling

### 4. Final Training
- Combined dataset: 682 clean + ~113 synthetic samples = ~795 total
- **~79% of test set utilized** through pseudo-labeling and clustering
- Ensemble of 10 models (5 initial + 5 final)
- Weighted ensemble using cross-validation scores

## Results & Performance

| Metric | Value |
|--------|-------|
| **Best Kaggle F1-Score** | **0.97** |
| **Dataset Expansion** | 682 + ~113 synthetic samples → ~795 total |
| **Unlabeled Data Utilization** | ~79% (113/144 test images) |
| **High-Confidence Pseudo-labels** | ~79 samples |
| **Clustered Core Samples** | ~34 samples |
| **Model Ensemble Size** | 10 models |

## Technical Implementation

### Architecture
- **Base Model**: Vision Transformer (ViT) via `timm` library, [`vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) variant
- **Classifier Head**: Custom head with Linear → BatchNorm → GELU → Dropout(0.4) → Linear
- **Weight Initialization**: Xavier Uniform for stable training
- **Loss Function**: Focal Loss (γ=2.0) with effective sample weighting and confidence weighting support

### Training Configuration
- **Optimizer**: AdamW with differential learning rates (backbone: 10% LR, head: full LR)
- **Scheduler**: CosineAnnealingWarmRestarts with warm restarts
- **Weight Decay**: 0.01 for weights, 0.0 for biases and normalization layers
- **Early Stopping**: Patience=5, min_delta=0.001
- **Data Augmentation**: Comprehensive Albumentations pipeline

### Key Innovations
- **Confidence-based filtering** prevents pseudo-label noise
- **Clustering purity checks** ensure high-quality synthetic samples
- **Weighted ensemble** balances clean vs. pseudo-labeled models
- **Effective class weighting** handles severe imbalance
- **Differential learning rates** for optimal fine-tuning

## Key Insights & Learnings

### What Worked
- **High confidence thresholds** (≥0.96) for pseudo-labeling
- **Clustering with purity checks** extracted valuable samples
- **Ensemble diversity** through different training strategies
- **Focal Loss + Effective Class Weights** handled imbalance effectively
- **Differential learning rates** for backbone vs. head optimization

### What Didn't Work
- Lower confidence thresholds introduced noise
- Blind pseudo-labeling without filtering
- Single model approaches
- Standard cross-entropy loss
- Uniform learning rates across backbone and head

### Best Practices Discovered
- **Quality over quantity** in synthetic data generation
- **Consistent feature space** for clustering effectiveness
- **Balanced ensemble weighting** for optimal performance
- **Robust data augmentation** for small datasets
- **Differential learning rates** for pre-trained model fine-tuning

## Resources

- **[GitHub Repository](https://github.com/ahmedsalim3/sheep-classification-challenge)**
- **[Kaggle Notebook](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model)**
- **[Competition Page](https://www.kaggle.com/competitions/sheep-classification-challenge-2025)**
- **[Discussion](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/discussion/585744)**

### Technical References
- **[Albumentations](https://albumentations.ai/)**
- **[*An Image is Worth 16x16 Words*](https://arxiv.org/pdf/2010.11929)**
- **[timm-model: vit_base_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)**
- **[UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/pdf/1802.03426)**
- **[Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555)**
- **[How to train your ViT? Data, Augmentation, and Regularization](https://arxiv.org/pdf/2106.10270)**
- **[Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)**


## License

This project is open source and available under the MIT License.

---

*Built with ❤️ for the Kaggle Sheep Classification Challenge 2025*
