KAGGLE_CREDS_PATH="/home/ahmedsalim/.kaggle/kaggle.json"
export KAGGLE_CONFIG_DIR=$(dirname "$KAGGLE_CREDS_PATH")
kaggle competitions submit -c sheep-classification-challenge-2025 -f output/submission.csv -m "message"
