import streamlit as st
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    AutoConfig
)

# --- Hugging Face Model Repo ---
MODEL_REPO = "amiguel/prototype_round3_qwen0.5-1.5B-merged"

# --- Page Configuration ---
st.set_page_config(
    page_title="DigiTwin - Qwen Advisor",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model(repo):
    # ✅ Override quantization_config to avoid bitsandbytes check
    config = AutoConfig.from_pretrained(repo)
    config.quantization_config = None  # Prevents looking for bitsandbytes

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32,  # Ensure safe for CPU inference
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model(MODEL_REPO)

# --- ChatML System Prompt ---
SYSTEM_PROMPT = (
    "You are DigiTwin, the digital twin of Ataliba, a senior inspection engineer with expertise in "
    "mechanical integrity, piping, and reliability. Answer professionally and concisely."
)

# --- Prompt Builder ---
def build_prompt(messages):
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    for msg in messages:
        role = msg["role"]
        prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# --- Generate Response ---
def generate_response(prompt_text):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    return streamer

# --- UI Layout ---
st.title("🧠 DigiTwin Qwen Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask your inspection question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    full_prompt = build_prompt(st.session_state.messages)
    with st.chat_message("assistant"):
        stream = generate_response(full_prompt)
        response_container = st.empty()
        full_response = ""
        for chunk in stream:
            full_response += chunk
            response_container.markdown(full_response + "▌", unsafe_allow_html=True)
        response_container.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
