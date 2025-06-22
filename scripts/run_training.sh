#!/bin/bash

# Training Runner Script
# This script runs training with different configurations (cross-validation or normal training)

set -e

echo "=========================================="
echo "Training Runner"
echo "=========================================="

# Default parameters
USE_CV=false
VAL_SPLIT=0.2
PSEUDO_LABELING=false
PSEUDO_THRESHOLD=0.95
CLUSTER_LABELING=false
PURITY_THRESHOLD=0.85

while [[ $# -gt 0 ]]; do
    case $1 in
        --cv)
            USE_CV=true
            shift
            ;;
        --normal)
            USE_CV=false
            shift
            ;;
        --val_split)
            VAL_SPLIT="$2"
            shift 2
            ;;
        --pseudo_labeling)
            PSEUDO_LABELING=true
            shift
            ;;
        --no_pseudo_labeling)
            PSEUDO_LABELING=false
            shift
            ;;
        --pseudo_threshold)
            PSEUDO_THRESHOLD="$2"
            shift 2
            ;;
        --cluster_labeling)
            CLUSTER_LABELING=true
            shift
            ;;
        --no_cluster_labeling)
            CLUSTER_LABELING=false
            shift
            ;;
        --purity_threshold)
            PURITY_THRESHOLD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cv                      Use cross-validation training"
            echo "  --normal                  Use normal training with validation split"
            echo "  --val_split RATIO         Validation split ratio for normal training (default: 0.2)"
            echo "  --pseudo_labeling         Enable pseudo-labeling"
            echo "  --no_pseudo_labeling      Disable pseudo-labeling"
            echo "  --pseudo_threshold T      Confidence threshold for pseudo-labeling (default: 0.95)"
            echo "  --cluster_labeling        Enable clustering-based labeling"
            echo "  --no_cluster_labeling     Disable clustering-based labeling"
            echo "  --purity_threshold T      Purity threshold for clustering (default: 0.85)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cv --pseudo_labeling                    # Cross-validation with pseudo-labeling"
            echo "  $0 --normal --val_split 0.1                 # Normal training with 10% validation"
            echo "  $0 --cv --pseudo_labeling --cluster_labeling # Full pipeline with clustering"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="uv run script/python3 -W ignore ./scripts/train.py"

if [ "$USE_CV" = true ]; then
    CMD="$CMD --use_cross_validation"
else
    CMD="$CMD --val_split $VAL_SPLIT"
fi

echo "Configuration:"
echo "  Training mode: $([ "$USE_CV" = true ] && echo "Cross-validation" || echo "Normal training")"
echo "  Validation split: $VAL_SPLIT"
echo "  Pseudo-labeling: $PSEUDO_LABELING"
echo "  Pseudo threshold: $PSEUDO_THRESHOLD"
echo "  Cluster labeling: $CLUSTER_LABELING"
echo "  Purity threshold: $PURITY_THRESHOLD"
echo ""

if [ "$PSEUDO_LABELING" = true ]; then
    export PSEUDO_LABELING=true
    export PSEUDO_THRESHOLD=$PSEUDO_THRESHOLD

    if [ "$CLUSTER_LABELING" = true ]; then
        export CLUSTER_LABELING=true
        export PURITY_THRESHOLD=$PURITY_THRESHOLD
    else
        export CLUSTER_LABELING=false
    fi
else
    export PSEUDO_LABELING=false
    export CLUSTER_LABELING=false
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "Training completed!"
echo "Results saved in: results/"
echo "  - Initial results: results/initial_results/"
echo "  - Predictions: results/predictions/"
if [ "$PSEUDO_LABELING" = true ]; then
    echo "  - Final results: results/final_results/"
fi
