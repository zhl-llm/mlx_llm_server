import streamlit as st
import time
import psutil
import os
from mlx_lm import load, stream_generate

st.set_page_config(page_title="MLX M4 LLM Server", page_icon="⚡️")
st.title("⚡️ MLX M4 LLM Server")

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# 1. Capture App Baseline
if "baseline_mem" not in st.session_state:
    st.session_state.baseline_mem = get_process_memory()

# 2. Load Model with Memory Tracking
@st.cache_resource
def init_model():
    start_mem = get_process_memory()
    model_path = "models/Qwen2.5-7B-Instruct"
    model, tokenizer = load(model_path, {"lazy": True})
    end_mem = get_process_memory()
    return model, tokenizer, (end_mem - start_mem)

model, tokenizer, model_weight_mem = init_model()

# --- Sidebar Setup ---
st.sidebar.header("💾 Memory Breakdown")
st.sidebar.info(f"🏠 App Baseline: {st.session_state.baseline_mem:.2f} GB")
st.sidebar.success(f"📦 Model Weights: {model_weight_mem:.2f} GB")
st.sidebar.header("⚙️ Assistant Configuration")
new_sys_prompt = st.sidebar.text_area("System Prompt", value="You are an accurate, professional, and concise AI assistant. Please answer users' questions in Chinese.")


st.sidebar.header("📊 Real-time Metrics")
ttft_metric = st.sidebar.empty()
tps_metric = st.sidebar.empty()
cache_mem_metric = st.sidebar.empty()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": new_sys_prompt}
    ]
else:
    st.session_state.messages[0]["content"] = new_sys_prompt

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create a clean placeholder for the streaming text
        response_placeholder = st.empty()
        full_response = ""
        
        # Build Context string
        formatted_prompt = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Performance Tracking
        start_time = time.time()
        ttft = None
        token_count = 0

        # --- CORRECT STREAMING LOOP ---
        # The key is iterating directly over stream_generate
        for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=1000):
            # 1. Handle Response Object (Supports both newer and older MLX versions)
            if hasattr(response, 'text'):
                delta_text = response.text
            else:
                delta_text = response # Fallback to string
            
            full_response += delta_text
            token_count += 1
            
            # 2. Capture TTFT on the very first token arrival
            if ttft is None and delta_text.strip() != "":
                ttft = time.time() - start_time
                ttft_metric.metric("First Token (TTFT)", f"{ttft*1000:.2f} ms")

            # 3. Calculate Performance (Update every token)
            elapsed = time.time() - start_time
            if ttft:
                # Speed = Tokens / Time since first token
                current_tps = token_count / (elapsed - ttft) if (elapsed - ttft) > 0 else 0
                tps_metric.metric("Speed", f"{current_tps:.2f} t/s")
            
            # 4. Update Memory Metrics
            dynamic_mem = (get_process_memory() - st.session_state.baseline_mem - model_weight_mem) * 1024
            cache_mem_metric.metric("Runtime/KV Cache", f"{max(0, dynamic_mem):.2f} MB")

            # 5. IMMEDIATELY update the UI for the "Streaming" effect
            response_placeholder.markdown(full_response + "▌")
        
        # Remove cursor at the end
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
