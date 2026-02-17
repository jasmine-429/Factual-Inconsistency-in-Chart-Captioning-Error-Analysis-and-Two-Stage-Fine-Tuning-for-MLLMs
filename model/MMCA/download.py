from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MAGAer13/mplug-owl-llama-7b",
    local_dir="/data/jguo376/pretrained_models/mplug-owl-llama-7b",
    local_dir_use_symlinks=False
)
