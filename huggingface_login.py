from huggingface_hub import HfApi

api = HfApi()
print(api.whoami())
