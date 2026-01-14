from huggingface_hub import snapshot_download

# set model ID and local dir for model storage
repo_id = "Qwen/Qwen-14B-Chat"
local_dir = "./models/qwen2.5-14b-chat"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # set to False to make sure the real model is stored, not softlink
    revision="main"
)
print(f"✅ model [ {repo_id} ] is downloaded successfully: {local_dir}")
