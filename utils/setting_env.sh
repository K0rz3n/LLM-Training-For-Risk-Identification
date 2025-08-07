# === Set environment variables, and rewrite everything to the data disk. ===

# The main directory for Hugging Face models and configurations (default ~/.cache/huggingface), move to the data drive.
export HF_HOME=/root/autodl-tmp/data/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/data/huggingface/hub

# The directory for local run logs/offline caching of Weights & Biases has been moved to the data drive.
export WANDB_DIR=/root/autodl-tmp/data/wandb_cache

# Create the four directories above one by one, along with an additional /root/autodl-tmp/data/checkpoints (used for storing model weights saved during training). The -p ensures no error is thrown if the directory already exists.
mkdir -p $HF_HOME $HF_HUB_CACHE $WANDB_DIR /root/autodl-tmp/data/checkpoints

echo "All catch dirs have been changed to /root/autodl-tmp/data/"
