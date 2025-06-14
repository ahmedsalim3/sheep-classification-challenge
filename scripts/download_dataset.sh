#!/bin/bash
# This script will download the raw Eid Al-Adha 2025: Sheep Classification Challenge dataset from Kaggle:
# https://www.kaggle.com/competitions/sheep-classification-challenge-2025/data,
# and save it in the 'data/' folder

# Ensure you have the Kaggle API key as described here:
# https://github.com/Kaggle/kaggle-api, and modify the
# KAGGLE_CREDS_PATH variable accordingly.

COMPETITION_NAME="sheep-classification-challenge-2025"
KAGGLE_CREDS_PATH="/home/ahmedsalim/.kaggle/kaggle.json"

if [ ! -f "$KAGGLE_CREDS_PATH" ]; then
    echo "Kaggle credentials file not found!"
    exit 1
fi

export KAGGLE_CONFIG_DIR=$(dirname "$KAGGLE_CREDS_PATH")
uv run kaggle competitions download -c "$COMPETITION_NAME" -p .
unzip -q sheep-classification-challenge-2025.zip && rm sheep-classification-challenge-2025.zip
mv "Sheep Classification Images"/* data && rm -rf "Sheep Classification Images"
