# mlx_llm_server

## Create virtual environment with pyenv

```sh
pyenv virtualenv 3.12.12 mlx-env
source ~/.pyenv/versions/mlx-env/bin/activate

pip install --upgrade pip
pip install huggingface_hub mlx mlx-lm fastapi streamlit uvicorn psutil tiktoken transformers modelscope
```

## Download models 

### Download models with huggigface

```sh
export HF_TOKEN=hf_XXXXXX
huggingface-cli download  Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b-instruct
```

### Download models with modelscope

```sh
modelscope download --model Qwen/Qwen-14B-Chat
```

## Download or convert models with mlx_lm

### Download quantization models with mlx_lm.convert

```sh
export HF_TOKEN=hf_XXXXXX

# Convert model with q4 quantization
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-14B \
  --mlx-path ./models/qwen2.5-14b-base-bits-4 \
  -q \
  --q-bits 4

# Convert model with q8 quantization 
mlx_lm.convert \
  --hf-path ./models/qwen2.5-14b-instruct \
  --mlx-path ./models/qwen2.5-14b-instruct-bits-8 \
  -q \
  --q-bits 8

# Convert model with key-layers are 6 bits and other layers are 4 bits
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-14B \
  --mlx-path ./models/qwen2.5-14b-base-mixed-4-6 \
  -q \
  --quant-predicate mixed_4_6

mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-72B \
  --mlx-path ./models/qwen2.5-72b-base-mixed-4-6 \
  -q \
  --quant-predicate mixed_4_6
```

### Start directly the LLM server with mlx_lm.server

```sh
python -m mlx_lm server --model /Users/zhlsunshine/Projects/inference/models/qwen2.5-14b-instruct-bits-8 --port 8080
```

### Start the LLM server with UI server

```sh
# set the limitation of united memory to 26GBi (26 * 1024 = 26624)
# will be default once the mac restart
sudo sysctl iogpu.wired_limit_mb=28672
streamlit run llm_model_server.py
```
