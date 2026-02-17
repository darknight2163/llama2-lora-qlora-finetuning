# Fine-Tuning LLaMA 2 using LoRA & QLoRA (PEFT)
## Project Overview
This project demonstrates how to fine-tune a large language model — LLaMA 2–7B Chat — using Parameter Efficient Fine-Tuning (PEFT) techniques such as:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
The training was performed on Google Colab (15GB GPU) using 4-bit quantization to reduce memory usage.

![Diagram comparing LLM pretraining, conventional fine-tuning, and parameter-efficient fine-tuning (PEFT) showing frozen base weights and small trainable adapters in PEFT.](https://github.com/darknight2163/llama2-lora-qlora-finetuning/blob/main/ft_peft.png)

## Why Fine-Tuning is Challenging?
Large models like **LLaMA 2–7B** (~7 Billion parameters) require:
- Massive GPU memory
- Storage for optimizer states
- Gradient memory
- Activation memory

Full fine-tuning would require ~40–60GB VRAM. Since Google Colab offers ~15GB GPU, full fine-tuning is not feasible.

## Why PEFT (LoRA / QLoRA)?
Instead of updating all model parameters, PEFT:

+ Freezes base model
+ Adds small trainable adapter layers
+ Updates only those adapters

This drastically reduces Memory usage, Training time, Compute cost.

## LoRA vs QLoRA
### LoRA
1. Adds low-rank matrices (A and B)
2. Injected into attention layers
3. Only small matrices are trained
4. Memory efficient but still loads model in FP16.

### QLoRA
QLoRA improves LoRA by:
1. Loading base model in 4-bit precision
2. Training LoRA adapters on top
3. Using NF4 quantization
4. This reduces memory usage by ~75%.

## Dataset used
* Original Dataset: timdettmers/openassistant-guanaco (https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftimdettmers%2Fopenassistant-guanaco)
* Reformatted Dataset: mlabonne/guanaco-llama2-1k (https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fmlabonne%2Fguanaco-llama2-1k)

### This dataset follows the LLaMA 2 chat template:
```<s>[INST] User prompt [/INST] Model answer```
### Why template matters?
Because LLaMA chat models expect structured conversation format.

## Model used
Base Model: NousResearch/Llama-2-7b-chat-hf
We trained: Llama-2-7b-chat-finetune

## Training Strategy
Key QLoRA Parameters

| Parameter | Value | Why |
| --- | --- | --- |
| r | 64 | Rank of low-rank matrices |
| alpha | 16 | Scaling factor |
| dropout | 0.1 | Regularization |
| quantization | 4-bit (NF4) | Memory saving |
| epochs | 1 | Demo training |
| optimizer | paged_adamw_32bit | Memory efficient |

## Training Pipeline
### Step 1: Install Required Packages

Uses:
-   transformers

-   peft

-   bitsandbytes

-   trl

-   accelerate

* * * * *

### Step 2: Load Dataset

We load preprocessed dataset from Hugging Face.

* * * * *

### Step 3: Configure 4-bit Quantization

`BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)`

Why NF4?

-   Normal Float 4 (better distribution for LLM weights)

-   More stable than FP4

* * * * *

### Step 4: Load Model in 4-bit

`AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)`

Base model weights are frozen.

* * * * *

### Step 5: Inject LoRA Layers

`LoraConfig(
    r=64,
    lora_alpha=16,
)`

LoRA layers are added to attention modules.
Only these layers are trainable.

* * * * *

### Step 6: Use SFTTrainer (Supervised Fine-Tuning)

We use `trl.SFTTrainer` which:

-   Handles tokenization

-   Applies prompt formatting

-   Runs training loop

* * * * *

### Step 7: Train

Only adapter weights are updated.

Base model remains unchanged.

* * * * *

### Step 8: Merge LoRA Weights

After training:

-   Reload base model in FP16

-   Merge LoRA weights

-   Save final model

Why?

Because LoRA stores only adapters.\
For deployment, we merge to create standalone model.

* * * * *

### Step 9: Push to Hugging Face Hub

Model uploaded for reuse.

* * * * *

## Monitoring Training

TensorBoard was used to monitor:
-   Loss curve
-   Learning rate schedule
* * * * *

## Inference Example

`prompt = "What is a large language model?"
pipe = pipeline(...)`
Input formatted as:
`<s>[INST] What is a large language model? [/INST]`

* * * * *

## Key Learnings
-   Full fine-tuning is expensive.
-   PEFT enables training large models on small GPUs.
-   QLoRA makes 7B models trainable on 15GB VRAM.
-   Prompt formatting is critical for chat models.
-   Merging adapters is necessary for deployment.

## Final Outcome
Successfully fine-tuned LLaMA 2–7B Chat using QLoRA on Google Colab 15GB GPU and deployed merged model to Hugging Face Hub.


















