from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    cache_dir="/root/autodl-tmp/data/huggingface/hub",
    resume_download=True, # Resume download
    local_dir_use_symlinks=False, # Turn off the symbolic link to prevent reference issues.
    max_workers=16
)
