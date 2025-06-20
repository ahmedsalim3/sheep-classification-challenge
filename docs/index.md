# Sheep Classification Challenge

A deep learning solution for the [Kaggle Sheep Classification Challenge 2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025/overview), achieving **0.97046 F1-score** using semi-supervised learning techniques on a small, imbalanced dataset.

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
- Vision Transformer (ViT) architecture
- Focal Loss + Class Weights for imbalance handling
- Custom training loop with dynamic learning rate scheduling

### 2. Pseudo-labeling
- Ensemble predictions on unlabeled test set
- **Strict confidence threshold (≥ 0.97)** for quality control
- Automatic filtering of low-confidence predictions

### 3. Clustering-based Data Mining
- **K-Means clustering** on ViT feature embeddings
- **Purity threshold (≥ 0.9)** for cluster filtering
- Extracted 34 high-quality samples from unlabeled data
- Feature space similarity for automatic labeling

### 4. Final Training
- Combined dataset: clean + pseudo-labeled + clustered samples
- **~115 total high confidence samples** (78% exposed of test unlabeled set)
- Ensemble of 10 models (5 initial + 5 final)
- Weighted ensemble using cross-validation scores

## Results & Performance

| Metric | Value |
|--------|-------|
| **Best Kaggle F1-Score** | **0.97046** |
| **Dataset Expansion** | 682 + ~115 synthetic samples → ~797 total |
| **Unlabeled Data Utilization** | 78% |
| **Clustered Samples Extracted** | 34 samples |
| **Model Ensemble Size** | 10 models |

## Technical Implementation

### Architecture
- **Base Model**: Vision Transformer (ViT) via `timm` library, [`vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) variant
- **Loss Function**: Focal Loss with effective sample weighting
- **Optimizer**: AdamW with dynamic learning rate scheduling
- **Data Augmentation**: Albumentations for robust training

### Key Innovations
- **Confidence-based filtering** prevents pseudo-label noise
- **Clustering purity checks** ensure high-quality synthetic samples
- **Weighted ensemble** balances clean vs. pseudo-labeled models
- **Effective class weighting** handles severe imbalance

## Project Structure

```
sheep-classification-challenge/
├── src/
│   ├── modeling/         # Training, evaluation, clustering
│   ├── data/             # Dataset, transforms, pseudo-labeling
│   └── utils/            # Helpers and utilities
├── scripts/
│   ├── train_cv.py       # Main training script
│   ├── workflow.py       # Complete pipeline orchestration
│   └── submit.sh         # Submission automation
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks
└── results/              # Training outputs and metrics
```

## Getting Started

### Installation
```bash
git clone https://github.com/ahmedsalim3/sheep-classification-challenge.git
cd sheep-classification-challenge
make install
```

### Quick Start
```bash
# Download dataset
make download-data

# Run complete pipeline
python scripts/train_cv.py

# Or use the workflow script
python scripts/workflow.py --mode full --use_clustering
```

### Configuration
Edit `configs/config.yml` to customize:
- Data paths and directories
- Training parameters
- Pseudo-labeling thresholds
- Clustering settings

## Key Insights & Learnings

### What Worked
- **High confidence thresholds** (≥0.97) for pseudo-labeling
- **Clustering with purity checks** extracted valuable samples
- **Ensemble diversity** through different training strategies
- **Focal Loss + Class Weights** handled imbalance effectively

### What Didn't Work
- Lower confidence thresholds introduced noise
- Blind pseudo-labeling without filtering
- Single model approaches
- Standard cross-entropy loss

### Best Practices Discovered
- **Quality over quantity** in synthetic data generation
- **Consistent feature space** for clustering effectiveness
- **Balanced ensemble weighting** for optimal performance
- **Robust data augmentation** for small datasets

## Resources

- **[GitHub Repository](https://github.com/ahmedsalim3/sheep-classification-challenge)**
- **[Kaggle Notebook](https://www.kaggle.com/code/ahvshim/fix-the-data-first-then-worry-about-the-model)**
- **[Competition Page](https://www.kaggle.com/competitions/sheep-classification-challenge-2025)**

- **[Albumentations](https://albumentations.ai/)**
- **[*An Image is Worth 16x16 Words*](https://arxiv.org/pdf/2010.11929)**
- **[timm-model: vit_base_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)**
- **[UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/pdf/1802.03426)**
- **[A Density-Based Algorithm for Discovering Clusters in Large Spatial Databaseswith Noise](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)**
- **[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294)**
- **[Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)**

## Contributing

Contributions are welcome! Feel free to:
- Submit pull requests
- Report issues
- Suggest improvements
- Share your own approaches

## License

This project is open source and available under the MIT License.

---

*Built with ❤️ for the Kaggle Sheep Classification Challenge 2025*
