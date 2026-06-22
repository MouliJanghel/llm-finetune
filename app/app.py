import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺")
st.title("Medical Q&A Chatbot")
st.caption("TinyLlama-1.1B fine-tuned on 2,000 medical Q&A pairs using LoRA")

col1, col2, col3 = st.columns(3)
col1.metric("Base ROUGE-L", "0.159")
col2.metric("Fine-tuned ROUGE-L", "0.161")
col3.metric("Improvement", "+1.3%")

st.divider()

@st.cache_resource
def load_model():
    base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    mdl = PeftModel.from_pretrained(mdl, "Moulijan/tinyllama-medical-finetuned")
    return mdl, tok

with st.spinner("Loading model — takes 1-2 mins first time..."):
    model, tokenizer = load_model()

st.success("Model ready! Ask a medical question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id
                )
            input_len = inputs["input_ids"].shape[1]
            ans = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        st.write(ans.strip())
        st.session_state.messages.append({"role": "assistant", "content": ans.strip()})

with st.sidebar:
    st.markdown("### About this project")
    st.markdown("""
- **Model:** TinyLlama-1.1B
- **Method:** LoRA fine-tuning (trained with 4-bit QLoRA on Colab T4 GPU)
- **Dataset:** 2,000 medical Q&A pairs
- **Deployment:** CPU inference on Hugging Face Spaces
- **[GitHub repo](https://github.com/MouliJanghel/llm-finetune-phi3)**
    """)
