# Hybrid ViT-CLIP Methodology

## Architectural Overview
The hybrid model combines two complementary computer vision architectures:

1. **Vision Transformer (ViT)**
   - Processes images as sequences of patches
   - Uses self-attention to model relationships between patches
   - Outputs feature representations from [CLS] token

2. **CLIP (Contrastive Language-Image Pretraining)**
   - Trained on image-text pairs with contrastive learning
   - Understands semantic relationships between images and text
   - Outputs normalized global image embeddings (frozen during training)

### Fusion Methods
We implement two feature fusion strategies:

#### 1. Concatenation Fusion

- Simple concatenation of ViT and CLIP features
- Preserves all information from both models
- Increases feature dimension (ViT_dim + CLIP_dim)

#### 2. Attention Fusion

- Uses multi-head self-attention mechanism
- Projects CLIP features to match ViT dimension
- Stacks ViT and CLIP features for joint attention
- Learns to weight and combine features dynamically
- Aggregates attended features through mean pooling

## Configuration Guide ([`configs/model.yml`](../configs/model.yml))

```yaml
# ViT Settings
vit:
  model_name: "google/vit-base-patch16-224-in21k"  # Alternatives:
                                                   # "google/vit-large-patch16-224"
                                                   # "google/vit-huge-patch14-224"

# CLIP Settings
clip:
  model_name: "ViT-B/32"  # Alternatives:
                           # "ViT-L/14" - Larger vision backbone
                           # "RN50x64" - CNN-based alternative

# Fusion Settings
fusion:
  method: "concat"  # Options:
                       # "concat" - Simple feature concatenation
                       # "attention" - Self-attention based fusion

# Training Settings
batch_size: 32       # Increase for larger GPUs
learning_rate: 2e-5  # Fine-tuning rate for ViT
epochs: 10           # Maximum epochs (early stopping may end sooner)
patience: 3          # Early stopping patience on val_f1_macro
val_split: 0.2       # Validation split ratio
seed: 42             # Reproducibility seed
```

### Models Reference

- [ViT Base - Patch16-224 (IN-21k)](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [ViT Large - Patch16-224](https://huggingface.co/google/vit-large-patch16-224)
- [ViT Huge - Patch14-224](https://huggingface.co/google/vit-huge-patch14-224)
- [CLIP ViT-B/32 (OpenAI)](https://huggingface.co/openai/clip-vit-base-patch32)
- [CLIP ViT-L/14 (OpenAI)](https://huggingface.co/openai/clip-vit-large-patch14)
- [CLIP RN50x64 (OpenAI)](https://huggingface.co/openai/clip-rn50x64)
- [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)
