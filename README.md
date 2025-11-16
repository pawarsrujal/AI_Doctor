## Fine-Tuning DeepSeek-R1 for Medical Chain-of-Thought Reasoning

---

A complete workflow to fine-tune the DeepSeek-R1 LLM on complex medical question-answering tasks using Unsloth for optimized performance on Colab with LoRA. This project is built to teach a model clinical reasoning and step-by-step diagnosis planning using a curated dataset.

**Table of Contents:**

---

Overview

Features

Model & Dataset

Training Pipeline

Technologies Used

Installation

Usage

Results

Future Work

License


**Overview:**

---

This project aims to fine-tune the DeepSeek-R1-Distill-Llama-8B model to answer complex clinical questions with step-by-step reasoning. It leverages the FreedomIntelligence/medical-o1-reasoning-SFT dataset, containing structured prompts with CoT (Chain-of-Thought) style explanations.**

**Features:**

---

Fine-tunes DeepSeek-R1 using Unsloth

Supports LoRA for efficient parameter training

Uses a structured medical CoT dataset

Inference-ready with clinical prompt styling

Integrated with Weights & Biases for experiment tracking

GPU-ready, bfloat16 and 4-bit quantization support

**Model & Dataset:**

---

Base Model
DeepSeek-R1-Distill-Llama-8B

Loaded using Unsloth for optimized inference and finetuning

Dataset
Source: FreedomIntelligence/medical-o1-reasoning-SFT

Structure:

Question: The medical case/question

Complex_CoT: Chain of Thought (reasoning)

Response: Final answer

Size: 500 training examples (sampled)

**Training Pipeline**

---

1. Environment Setup
   
Install Unsloth and dependencies

Check GPU availability

Login to HuggingFace and Weights & Biases using tokens

2. Load Model
 
```python
model, tokenizer = FastLanguageModel.from_pretrained(

    model_name = "dee/DeepSeek-R1-Distill-Llama-8B",
   
    max_seq_length = 2048,
   
    load_in_4bit = True,
   
    token = hf_token
)

```

3. Prompt Template

    Task:
You are a medical expert specializing in clinical reasoning...

```python

 Query:
{question}

 Answer:
{output}


```

4. Inference Test (Before Finetuning)
   
Run sample queries through the base model to establish baseline

5. Dataset Preparation
Format each sample into:

```python

 Question:
<question>

 Response:
<Chain-of-thought>

<Final answer>
Append <eos> token for training

```

6. LoRA Integration

   
Applied to attention projection layers:

```python

q_proj, k_proj, v_proj, o_proj, etc.

Training with TRL’s SFTTrainer
Batch size: 2

Epochs: 1

Accumulation steps: 4

Max steps: 60

Optimizer: adamw_8bit

Mixed precision: bfloat16 or fp16

```


**Technologies Used**	

---

Unsloth           	

HuggingFace       	

DeepSeek-R1	        

TRL (SFTTrainer)	  

Colab / CUDA	      

W&B	               

PyTorch


**How to Run**
---

1. Clone the Repo
   
```bash
git clone https://github.com/yourusername/ai-doctor.git
cd ai-doctor
```

2. Install Dependencies

```bash
pip install torch transformers datasets trl wandb unsloth accelerate streamlit
```

3. Train the Model
Edit and run train.py to fine-tune DeepSeek-R1:

```bash
python train.py
 GPU with ≥12GB VRAM is required. Unsloth only supports NVIDIA GPUs.
```

4. Run Inference

```bash
python infer.py
```

5. Launch the Streamlit App

```bash
streamlit run app.py
```

**Future Work**

---

Train on full dataset (10k+ samples)

Incorporate medical textbooks (PubMed, UMLS)

Add evaluation using medical QA benchmarks (MedQA, USMLE)

Serve via FastAPI or Streamlit for frontend
