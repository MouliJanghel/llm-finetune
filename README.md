# LLM Fine-Tuning: TinyLlama on Medical Q&A

Fine-tuned **TinyLlama-1.1B-Chat** on 2,000 medical Q&A examples using
**LoRA (rank-8, 4-bit QLoRA)** via PEFT — achieving a measured **+1.3%
improvement in ROUGE-L** over the base model on a 100-example held-out
test set. Trained on **Google Colab free T4 GPU** in under 20 minutes.

## 🔗 Live Demo
👉 [Try the chatbot here](https://huggingface.co/spaces/Moulijan/tinyllama-medical-demo)**

## 📊 Results

| Model              | ROUGE-L |
|--------------------|---------|
| TinyLlama (base)   | 0.159   |
| Fine-tuned (mine)  | 0.161   |
| **Improvement**    | **+1.3%** |

## 🛠️ Tech Stack
`Python` `PyTorch` `Hugging Face Transformers` `PEFT/LoRA` `TRL`
`BitsAndBytes (4-bit)` `Weights & Biases` `Streamlit`

## 🏗️ Architecture & Training
- **Base model:** TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning:** QLoRA (rank=8, alpha=16, 4-bit quantization)
- **Dataset:** ChatDoctor-HealthCareMagic (2,000 training examples)
- **Training:** Google Colab T4 GPU (free), ~20 min, 3 epochs
- **Evaluation:** ROUGE-1/ROUGE-L on 100 held-out examples
- **Deployment:** CPU inference on Hugging Face Spaces (Streamlit)

## 📝 Notes
First end-to-end LLM fine-tuning project. Built the complete pipeline:
data preprocessing, LoRA fine-tuning, quantitative evaluation, and live
deployment — entirely on free-tier infrastructure.
