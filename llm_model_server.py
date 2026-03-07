import streamlit as st
import time
import psutil
import os
import mlx.core as mx  # 用于监控 Apple Silicon 显存
from mlx_lm import load, stream_generate

st.set_page_config(page_title="MLX M4 LLM 服务器", page_icon="⚡️")
st.title("⚡️ MLX M4 LLM 推理服务器")
st.sidebar.header("⚙️ 助手配置")
new_sys_prompt = st.sidebar.text_area("系统提示词 (System Prompt)", value="你是一个专业的 你是一个专业的AI辅助机器人，请认真且准确地回答问题。")

st.sidebar.header("📈 实时性能指标")
ttft_metric = st.sidebar.empty() # 首个 Token 延迟
tps_metric = st.sidebar.empty()  # 生成速度 (token/s)
token_count_metric = st.sidebar.empty()  # 已生成的 Token 数

def get_process_memory():
    """获取当前进程占用的物理内存 (GB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# 1. 记录应用启动时的基准内存
if "baseline_mem" not in st.session_state:
    st.session_state.baseline_mem = get_process_memory()

# 带有内存追踪的模型加载函数
@st.cache_resource
def init_model():
    # 1. 记录应用启动基准
    baseline = get_process_memory()

    model_path = "models/MiniCPM4.1-8B-MLX"
    model, tokenizer = load(model_path, {"lazy": True})

    # 强制评估所有参数，确保权重真正占用显存，否则权重会被计入后续的 KV Cache 中
    mx.eval(model.parameters())

    # 2. 此时获取的活动显存即为“纯静态权重”，使用推荐的新版 API: mx.get_active_memory()
    total_active_now = mx.get_active_memory() / (1024 ** 3)

    return model, tokenizer, total_active_now, model_path

model, tokenizer, model_weight_mem, model_path = init_model()
st.sidebar.markdown(f"**Model:** {model_path}")

# --- 侧边栏设置 (内存指标整合) ---
st.sidebar.header("📊 内存动态监控")
# 按行创建容器
with st.sidebar.container():
    sys_mem_metric = st.sidebar.empty()    # 第一行：系统总占用
    mlx_active_metric = st.sidebar.empty() # 第二行：MLX 活动显存
    cache_mem_metric = st.sidebar.empty()  # 第三行：KV Cache 占用
    st.sidebar.divider()

# --- 动态更新函数 ---
# 系统进程内存 (RSS) ≈ 活动显存 (Active Memory) + 应用程序开销 (Python/Streamlit)
# 活动显存 (Active Memory) = 模型静态权重 (Weights) + 动态缓存 (KV Cache)
def update_metrics(token_count):
    # A. 获取当前 MLX 总活动内存 (GB)
    # 包含：静态权重 + 动态 KV Cache
    current_active_gb = mx.get_active_memory() / (1024 ** 3)

    # B. 计算 KV Cache (GB)
    # 现在的逻辑是：(权重 + Cache) - 权重 = Cache
    kv_cache_gb = max(0, current_active_gb - model_weight_mem)

    # C. 获取系统进程内存 (作为参考)
    current_rss = get_process_memory()

    ########################################################################
    ### “系统进程内存 (RSS)” 会远小于“活动显存”。这是因为 Apple Metal 框架分配的内存
    ### 会被归类为“系统驱动层占用”，而不完全计入“用户进程私有占用”。
    ########################################################################
    # 1. 系统进程内存：反映 macOS 报告的物理占用
    sys_mem_metric.metric("🖥️ 系统进程内存 (RSS)", f"{current_rss:.3f} GB")

    # 2. 活动显存：反映 GPU 实际锁定的统一内存（最准）
    mlx_active_metric.metric("🧠 实际活动显存", f"{current_active_gb:.3f} GB")

    # 3. KV Cache：显示高精度数值或转为 MB
    if kv_cache_gb < 0.5:
        cache_mem_metric.metric("🌀 KV Cache 占用", f"{kv_cache_gb * 1024:.2f} MB")
    else:
        cache_mem_metric.metric("🌀 KV Cache 占用", f"{kv_cache_gb:.4f} GB")

    token_count_metric.metric("🔢 已生成计数", f"{token_count} tokens")

# --- 聊天逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": new_sys_prompt}]
else:
    st.session_state.messages[0]["content"] = new_sys_prompt

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("在此输入消息..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 构建上下文
        formatted_prompt = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 性能追踪初始化
        start_time = time.time()
        ttft = None
        token_count = 0

        # --- 流式生成循环 ---
        for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=102400):
            # 1. 处理响应对象 (兼容不同 MLX 版本)
            delta_text = response.text if hasattr(response, 'text') else response
            
            full_response += delta_text
            token_count += 1
            
            # 2. 记录首个 Token 到达时间 (TTFT)
            if ttft is None and delta_text.strip() != "":
                ttft = time.time() - start_time
                generation_start_time = time.time() # 从这一刻开始算 TPS
                ttft_metric.metric("🚀 首字延迟", f"{ttft*1000:.0f} ms")

            # 3. 计算生成速度 (TPS)
            if generation_start_time:
                t_elapsed = time.time() - generation_start_time
                if t_elapsed > 0:
                    tps_metric.metric("⚡️ 生成速度", f"{token_count / t_elapsed:.2f} t/s")

            # 4. 更新 Token 统计
            token_count_metric.metric("已生成 Token", f"{token_count} tokens")
            
            # 5. 实时更新侧边栏内存数据
            update_metrics(token_count)

            # 6. 更新 UI 渲染
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
