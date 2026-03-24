# From Base Model to ChatGPT: Instruction Tuning & Alignment with superGPT

**How to turn a raw text-completion model into an AI that answers questions, follows instructions, and reasons step-by-step — just like ChatGPT, Claude, and DeepSeek.**

---

## Table of Contents

1. [The 4 Stages of Building a Chat AI](#1-the-4-stages-of-building-a-chat-ai)
2. [Stage 1: Pre-Training (Base Model)](#2-stage-1-pre-training-base-model)
3. [Stage 2: Supervised Fine-Tuning (SFT)](#3-stage-2-supervised-fine-tuning-sft)
4. [Stage 3: Preference Alignment (DPO)](#4-stage-3-preference-alignment-dpo)
5. [Stage 4: Reinforcement Learning (RLHF / RLVR)](#5-stage-4-reinforcement-learning-rlhf--rlvr)
6. [Instruction Datasets](#6-instruction-datasets)
7. [Chat Templates & Special Tokens](#7-chat-templates--special-tokens)
8. [Complete Pipeline: Base → Chat in 4 Commands](#8-complete-pipeline-base--chat-in-4-commands)
9. [Serving Your Chat Model](#9-serving-your-chat-model)
10. [Reasoning Models (DeepSeek-R1 Style)](#10-reasoning-models-deepseek-r1-style)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Real-World Examples](#12-real-world-examples)
13. [FAQ & Troubleshooting](#13-faq--troubleshooting)

---

## 1. The 4 Stages of Building a Chat AI

Every chat AI you've ever used — ChatGPT, Claude, Gemini, DeepSeek — was built in the same 4 stages:

```
Stage 1                Stage 2              Stage 3              Stage 4
─────────────────── → ─────────────────── → ─────────────────── → ───────────────────
PRE-TRAINING          SFT                  DPO / RLHF           RLVR / Safety
                      (Supervised           (Preference           (Reinforcement
  Raw text             Fine-Tuning)         Alignment)            Learning)
  completion
                      Learns to follow     Learns to prefer     Learns to reason
  "The cat sat"       instructions         good answers over    and self-verify
  → "on the mat"      Q → A format         bad ones
                                                                 <think>...</think>
  ↓                   ↓                    ↓                    ↓
  Base Model          Instruct Model       Aligned Model        Reasoning Model
  (useless alone)     (follows orders)     (actually helpful)   (thinks step-by-step)
```

### What Each Stage Does

| Stage | What It Learns | Data Needed | superGPT Module |
|-------|---------------|-------------|-----------------|
| **1. Pre-Training** | Language patterns, facts, grammar | Trillions of tokens of raw text | `train.py` |
| **2. SFT** | How to follow instructions, Q&A format | ~100K instruction/response pairs | `finetune.py` |
| **3. DPO** | Prefer good answers, reject bad ones | ~50K preference pairs (chosen vs rejected) | `align.py` |
| **4. RLHF/RLVR** | Self-improvement, reasoning, safety | Reward model or verifiable tasks | `rlhf.py` / `rlvr.py` |

### How Much Each Stage Costs

| Stage | Data Size | Time (A100) | Cost |
|-------|-----------|-------------|------|
| Pre-training | 100M-1T tokens | Hours to weeks | $1 - $100K |
| SFT | 50K-500K examples | 1-4 hours | $2-7 |
| DPO | 20K-100K pairs | 1-3 hours | $2-5 |
| RLHF/RLVR | 1K-10K prompts | 2-8 hours | $3-13 |

> **Key insight:** Pre-training is expensive. Everything after it is cheap. You can turn a $50 base model into a ChatGPT-like assistant for under $25 of additional compute.

---

## 2. Stage 1: Pre-Training (Base Model)

This is what we've been doing — training a model on raw text to learn language. If you've already trained a model using the [Getting Started guide](getting-started.md), you have a base model checkpoint at `checkpoints/best.pt`.

**What a base model can do:** Continue text patterns.  
**What it can't do:** Answer questions, follow instructions, have conversations.

```bash
# Example: Base model behavior
$ python -m supergpt.inference.generate --checkpoint checkpoints/best.pt \
    --prompt "What is the capital of France?"

# Output (base model): "What is the capital of France? This is a common
# geography question that appears on many standardized tests. The answer..."
# ↑ It doesn't ANSWER — it continues writing as if it's in a textbook!
```

To make it actually answer questions, we need Stage 2: SFT.

---

## 3. Stage 2: Supervised Fine-Tuning (SFT)

SFT teaches the model the **format** of a conversation: when given a question, produce an answer.

### 3.1 Preparing Instruction Data

You need data in prompt→completion format. Here are the main formats:

#### Format A: Simple Prompt-Completion (JSONL)

```json
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris. It is located in the north-central part of the country along the Seine River."}
{"prompt": "Write a Python function to reverse a string.", "completion": "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```\n\nThis function uses Python's slice notation with a step of -1 to reverse the string."}
{"prompt": "Explain quantum computing in simple terms.", "completion": "Quantum computing is a type of computing that uses quantum mechanical phenomena like superposition and entanglement to process information. While classical computers use bits (0 or 1), quantum computers use qubits that can be both 0 and 1 simultaneously. This allows them to solve certain problems exponentially faster."}
```

#### Format B: Chat-Style with Roles (JSONL)

```json
{"messages": [{"role": "system", "content": "You are a helpful math tutor."}, {"role": "user", "content": "What is 15 + 27?"}, {"role": "assistant", "content": "15 + 27 = 42.\n\nHere's how I solved it:\n- 15 + 27\n- 15 + 20 = 35\n- 35 + 7 = 42"}]}
{"messages": [{"role": "user", "content": "Translate 'hello' to French"}, {"role": "assistant", "content": "The French translation of 'hello' is 'bonjour'."}]}
{"messages": [{"role": "system", "content": "You are a code reviewer."}, {"role": "user", "content": "Review this code:\ndef add(a,b): return a+b"}, {"role": "assistant", "content": "The function works but could be improved:\n1. Add type hints: `def add(a: int, b: int) -> int:`\n2. Add a docstring explaining what it does\n3. Consider input validation"}]}
```

#### Format C: Alpaca Format (common in the community)

```json
{"instruction": "Summarize the following text.", "input": "The Industrial Revolution was a period of major mechanization...", "output": "The Industrial Revolution was a transformative era (1760-1840) that shifted economies from agrarian to industrial, beginning in Britain with innovations in textile manufacturing, steam power, and iron production."}
{"instruction": "Write a haiku about programming.", "input": "", "output": "Code flows like water\nBugs emerge from the shadows\nTests bring peace of mind"}
```

### 3.2 Where to Get Instruction Data

#### Option 1: Use Existing High-Quality Datasets

```bash
# Download popular instruction datasets from HuggingFace:

# OpenHermes 2.5 — 1M high-quality instructions (BEST for general-purpose)
pip install datasets
python -c "
from datasets import load_dataset
import json

ds = load_dataset('teknium/OpenHermes-2.5', split='train[:100000]')

with open('sft_data/instructions.jsonl', 'w') as f:
    for item in ds:
        conv = item['conversations']
        if len(conv) >= 2:
            prompt = conv[0]['value'] if conv[0]['from'] == 'human' else ''
            response = conv[1]['value'] if conv[1]['from'] == 'gpt' else ''
            if prompt and response:
                f.write(json.dumps({'prompt': prompt, 'completion': response}) + '\n')

print('Done! Check sft_data/instructions.jsonl')
"

# SlimOrca — 500K curated instruction-response pairs
python -c "
from datasets import load_dataset
import json

ds = load_dataset('Open-Orca/SlimOrca', split='train[:50000]')

with open('sft_data/slim_orca.jsonl', 'w') as f:
    for item in ds:
        convs = item['conversations']
        prompt = next((c['value'] for c in convs if c['from'] == 'human'), '')
        response = next((c['value'] for c in convs if c['from'] == 'gpt'), '')
        if prompt and response:
            f.write(json.dumps({'prompt': prompt, 'completion': response}) + '\n')
"

# Dolly 15K — Databricks open-source instruction data
python -c "
from datasets import load_dataset
import json

ds = load_dataset('databricks/databricks-dolly-15k', split='train')

with open('sft_data/dolly.jsonl', 'w') as f:
    for item in ds:
        prompt = item['instruction']
        if item['context']:
            prompt += f'\n\nContext: {item[\"context\"]}'
        f.write(json.dumps({
            'prompt': prompt,
            'completion': item['response']
        }) + '\n')
"
```

#### Option 2: Code-Specific Instructions

```bash
# Code Alpaca — 20K code instruction pairs
python -c "
from datasets import load_dataset
import json

ds = load_dataset('sahil2801/CodeAlpaca-20k', split='train')

with open('sft_data/code_instructions.jsonl', 'w') as f:
    for item in ds:
        prompt = item['instruction']
        if item.get('input'):
            prompt += f'\n\nInput: {item[\"input\"]}'
        f.write(json.dumps({
            'prompt': prompt,
            'completion': item['output']
        }) + '\n')
"

# Magicoder — 75K synthetic code instructions (GPT4-generated)
python -c "
from datasets import load_dataset
import json

ds = load_dataset('ise-uiuc/Magicoder-Evol-Instruct-110K', split='train[:50000]')

with open('sft_data/magicoder.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({
            'prompt': item['instruction'],
            'completion': item['response']
        }) + '\n')
"
```

#### Option 3: Math & Reasoning Instructions

```bash
# MetaMathQA — 395K math questions with solutions
python -c "
from datasets import load_dataset
import json

ds = load_dataset('meta-math/MetaMathQA', split='train[:100000]')

with open('sft_data/math_instructions.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({
            'prompt': item['query'],
            'completion': item['response']
        }) + '\n')
"
```

#### Option 4: Create Your Own Data

```python
# generate_instructions.py — Create custom instruction data

import json

# Write your domain-specific Q&A pairs
custom_data = [
    {
        "prompt": "What are the main features of superGPT?",
        "completion": "superGPT is a production-grade LLM training framework with:\n"
                     "1. Multi-Head Latent Attention (MLA) from DeepSeek-V3\n"
                     "2. Mixture of Experts (MoE) with up to 64 experts\n"
                     "3. Multi-Token Prediction (MTP)\n"
                     "4. LoRA fine-tuning for efficient adaptation\n"
                     "5. FSDP multi-GPU training\n"
                     "6. FP8 training on H100 GPUs"
    },
    {
        "prompt": "How do I train a model with superGPT?",
        "completion": "To train a model with superGPT:\n\n"
                     "1. Prepare data: `python -m supergpt.training.data_pipeline "
                     "--dataset HuggingFaceFW/fineweb-edu --output-dir data/`\n\n"
                     "2. Train: `python -m supergpt.training.train --preset small "
                     "--data-dir data/ --compile --device cuda`\n\n"
                     "3. Generate: `python -m supergpt.inference.generate "
                     "--checkpoint checkpoints/best.pt --prompt 'Hello'`"
    },
    # Add hundreds more for your domain...
]

with open('sft_data/custom.jsonl', 'w') as f:
    for item in custom_data:
        f.write(json.dumps(item) + '\n')
```

### 3.3 Converting Instruction Data to Training Format

superGPT's `finetune.py` uses tokenized binary files (`.bin`). You need to convert your JSONL instruction data into this format:

```python
#!/usr/bin/env python3
"""Convert instruction JSONL to tokenized binary format for superGPT SFT."""

import json
import os
import pickle
import numpy as np

def convert_instructions_to_binary(
    input_jsonl: str,
    output_dir: str,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    val_fraction: float = 0.05,
    chat_template: str = "chatml",  # "chatml", "llama", "simple"
):
    """Convert instruction JSONL to tokenized train.bin + val.bin."""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer: {tokenizer_name} (vocab: {tokenizer.vocab_size:,})")

    # Define chat template
    def format_example(item):
        prompt = item.get("prompt", item.get("instruction", ""))
        completion = item.get("completion", item.get("output", item.get("response", "")))

        if chat_template == "chatml":
            # ChatML format (used by Qwen, many open models)
            return (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{completion}<|im_end|>\n"
            )
        elif chat_template == "llama":
            # Llama-3 format
            return (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{completion}<|eot_id|>"
            )
        else:
            # Simple format
            return f"### Instruction:\n{prompt}\n\n### Response:\n{completion}\n\n"

    # Load and tokenize
    all_tokens = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = format_example(item)
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

    print(f"Total tokens: {len(all_tokens):,}")

    # Split train/val
    all_tokens = np.array(all_tokens, dtype=np.uint32)
    split_idx = int(len(all_tokens) * (1 - val_fraction))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save
    train_tokens.tofile(os.path.join(output_dir, "train.bin"))
    val_tokens.tofile(os.path.join(output_dir, "val.bin"))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": "tiktoken",
        "tokenizer_name": tokenizer_name,
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved to {output_dir}/")
    print(f"  train.bin: {len(train_tokens):,} tokens ({train_tokens.nbytes/1e6:.1f} MB)")
    print(f"  val.bin:   {len(val_tokens):,} tokens ({val_tokens.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sft_data/instructions.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data_sft/"
    convert_instructions_to_binary(input_file, output_dir)
```

### 3.4 Running SFT with LoRA

Now fine-tune your base model on the instruction data:

```bash
# LoRA fine-tuning (memory efficient — works on 8GB GPU!)
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data data_sft/ \
    --lora-rank 64 \
    --lora-alpha 16 \
    --max-iters 5000 \
    --lr 2e-5 \
    --batch-size 8 \
    --compile \
    --device cuda
```

### LoRA Rank Guide

| Rank | Trainable Params | Quality | Best For |
|------|-----------------|---------|----------|
| 4-8 | ~0.1% | Decent | Quick experiments, testing |
| 16-32 | ~0.5% | Good | Domain adaptation |
| 64 | ~1.5% | Great | General instruction-following |
| 128-256 | ~3-5% | Excellent | Maximum quality SFT |

### 3.5 Testing Your Instruction-Tuned Model

```bash
# Now it should ANSWER questions instead of just continuing text:
python -m supergpt.inference.generate \
    --checkpoint checkpoints/finetune_best.pt \
    --prompt "What is the capital of France?" \
    --max-tokens 200 \
    --temperature 0.7

# Expected output (after SFT):
# "The capital of France is Paris. It is located in the
#  north-central part of the country along the Seine River..."
# ↑ It ANSWERS! This is the magic of SFT.
```

---

## 4. Stage 3: Preference Alignment (DPO)

After SFT, your model follows instructions but might give mediocre answers. DPO teaches it to **prefer good answers over bad ones**.

### 4.1 What is DPO?

DPO (Direct Preference Optimization) is the modern replacement for RLHF. Instead of training a separate reward model and using PPO, DPO directly optimizes the policy from preference pairs.

```
Traditional RLHF:  SFT Model → Reward Model → PPO Training → Aligned Model
                                (complex, unstable)

DPO:               SFT Model → Preference Data → Direct Optimization → Aligned Model
                                (simple, stable, same results!)
```

### 4.2 Preparing Preference Data

Each example has: a **prompt**, a **chosen** (good) response, and a **rejected** (bad) response.

```json
{"prompt": "What is 2+2?", "chosen": "2 + 2 = 4.", "rejected": "2 + 2 = 5. This is because when you add two and two together, you need to carry the one."}
{"prompt": "Write a Python function to sort a list.", "chosen": "```python\ndef sort_list(lst):\n    return sorted(lst)\n```\nThis uses Python's built-in `sorted()` function which implements Timsort (O(n log n)).", "rejected": "Here is a sort function:\ndef sort(l):\n  for i in range(len(l)):\n    for j in range(len(l)):\n      if l[i] < l[j]:\n        l[i], l[j] = l[j], l[i]\n  return l"}
{"prompt": "Is the Earth flat?", "chosen": "No, the Earth is not flat. It is an oblate spheroid — slightly flattened at the poles and bulging at the equator. This has been confirmed through satellite imagery, GPS systems, physics experiments, and centuries of scientific observation.", "rejected": "Some people believe the Earth is flat, and there are interesting arguments on both sides of the debate."}
```

#### Where to Get Preference Data

```bash
# Anthropic HH-RLHF — 170K human preference pairs
python -c "
from datasets import load_dataset
import json

ds = load_dataset('Anthropic/hh-rlhf', split='train[:50000]')

with open('dpo_data/preferences.jsonl', 'w') as f:
    for item in ds:
        # Extract prompt from conversation
        chosen = item['chosen']
        rejected = item['rejected']

        # Find the last human turn as prompt
        parts = chosen.split('\n\nHuman: ')
        if len(parts) >= 2:
            prompt = parts[-1].split('\n\nAssistant: ')[0]
            chosen_resp = parts[-1].split('\n\nAssistant: ')[-1] if '\n\nAssistant: ' in parts[-1] else ''
            rejected_parts = rejected.split('\n\nAssistant: ')
            rejected_resp = rejected_parts[-1] if len(rejected_parts) > 1 else ''

            if prompt and chosen_resp and rejected_resp:
                f.write(json.dumps({
                    'prompt': prompt.strip(),
                    'chosen': chosen_resp.strip(),
                    'rejected': rejected_resp.strip()
                }) + '\n')

print('Done!')
"

# UltraFeedback — 64K AI-rated preference pairs (used by Zephyr)
python -c "
from datasets import load_dataset
import json

ds = load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='train_prefs[:50000]')

with open('dpo_data/ultrafeedback.jsonl', 'w') as f:
    for item in ds:
        prompt = item['prompt']
        chosen = item['chosen'][1]['content'] if len(item['chosen']) > 1 else ''
        rejected = item['rejected'][1]['content'] if len(item['rejected']) > 1 else ''
        if prompt and chosen and rejected:
            f.write(json.dumps({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            }) + '\n')
"
```

### 4.3 Running DPO Alignment

```bash
# Align your SFT model with preference data
python -m supergpt.alignment.align \
    --checkpoint checkpoints/finetune_best.pt \
    --data dpo_data/preferences.jsonl \
    --beta 0.1 \
    --max-iters 2000 \
    --lr 5e-6 \
    --device cuda
```

### DPO Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Beta | `--beta` | 0.1 | Alignment strength. Higher = stronger preference, but less diverse. 0.05-0.5 typical |
| Learning rate | `--lr` | 5e-6 | Keep very low! DPO is a fine-grained adjustment |
| Iterations | `--max-iters` | 2000 | Usually 1000-5000 is sufficient |

### What DPO Actually Does (The Math)

```
DPO Loss = -log σ(β × (log π(chosen|prompt) - log π_ref(chosen|prompt)
                       - log π(rejected|prompt) + log π_ref(rejected|prompt)))

In English:
- π = your model (being trained)
- π_ref = frozen copy of your model before DPO
- It makes your model MORE likely to generate "chosen" responses
- And LESS likely to generate "rejected" responses
- β controls how much this matters (higher = stronger preferences)
```

---

## 5. Stage 4: Reinforcement Learning (RLHF / RLVR)

This is the final polish. superGPT supports three RL methods:

### 5.1 PPO (Classic RLHF)

The original method used by ChatGPT. Requires 4 models running simultaneously:

```bash
# Step 1: Train a reward model from preference data
python -m supergpt.alignment.rlhf reward \
    --checkpoint checkpoints/best.pt \
    --data dpo_data/preferences.jsonl \
    --output reward_model.pt

# Step 2: Run PPO with the reward model
python -m supergpt.alignment.rlhf ppo \
    --checkpoint checkpoints/dpo_aligned.pt \
    --reward-model reward_model.pt \
    --max-steps 1000 \
    --device cuda
```

### 5.2 GRPO (DeepSeek Style — Simpler & Better)

Group Relative Policy Optimization — no value model needed (only 2 models instead of 4):

```bash
# GRPO with a reward model
python -m supergpt.alignment.rlhf grpo \
    --checkpoint checkpoints/dpo_aligned.pt \
    --reward-model reward_model.pt \
    --group-size 4 \
    --max-steps 1000

# GRPO with rule-based rewards (no reward model at all!)
python -m supergpt.alignment.rlhf grpo \
    --checkpoint checkpoints/dpo_aligned.pt \
    --rule-reward length \
    --max-steps 500
```

### 5.3 RLVR — Reinforcement Learning with Verifiable Rewards (DeepSeek-R1 Style)

The most exciting method. **No human labels needed** — the model learns to reason using auto-verifiable tasks:

```bash
# Train math reasoning (answers are verifiable!)
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/dpo_aligned.pt \
    --task math \
    --data math_prompts.jsonl \
    --require-format \
    --format-spec cot \
    --max-steps 500 \
    --device cuda

# Train code generation (execute and verify!)
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/dpo_aligned.pt \
    --task code \
    --data code_prompts.jsonl \
    --max-steps 500
```

#### RLVR Data Format

```json
{"prompt": "What is 15 + 27?", "answer": "42"}
{"prompt": "What is the square root of 144?", "answer": "12"}
{"prompt": "If x + 3 = 7, what is x?", "answer": "4"}
{"prompt": "What is 2^10?", "answer": "1024"}
```

For code tasks:
```json
{"prompt": "Write a function that returns the nth Fibonacci number.", "test_code": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55\nassert fib(20) == 6765"}
```

#### What Makes RLVR Special

This is exactly how DeepSeek built R1, their reasoning model. The key insight:

> **Pure RL with verifiable rewards produces emergent reasoning capabilities** — the model spontaneously develops chain-of-thought, self-reflection, and error-correction without being explicitly taught these behaviors.

superGPT's RLVR includes three verifiers:

| Verifier | What It Checks | Supports |
|----------|---------------|----------|
| **MathVerifier** | Numeric answers | `\boxed{}`, `####`, "the answer is", last number |
| **CodeVerifier** | Code execution | Python code extraction, sandboxed execution, assertions |
| **FormatVerifier** | Output structure | `<think>...</think>`, step format, JSON validity |

---

## 6. Instruction Datasets

Here's a comprehensive list of the best instruction datasets available:

### General Purpose

| Dataset | Size | Quality | Best For |
|---------|------|---------|----------|
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1M | ⭐⭐⭐⭐⭐ | Best overall general SFT |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 518K | ⭐⭐⭐⭐ | High quality, curated |
| [Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15K | ⭐⭐⭐ | Small but human-written |
| [LIMA](https://huggingface.co/datasets/GAIR/lima) | 1K | ⭐⭐⭐⭐⭐ | Tiny but extremely high quality |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara) | 16K | ⭐⭐⭐⭐ | Multi-turn conversations |

### Code

| Dataset | Size | Quality | Best For |
|---------|------|---------|----------|
| [Magicoder-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | 110K | ⭐⭐⭐⭐⭐ | Code generation |
| [Code-Alpaca-20K](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 20K | ⭐⭐⭐ | Quick code SFT |
| [CodeFeedback](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 157K | ⭐⭐⭐⭐ | Code with feedback |

### Math & Reasoning

| Dataset | Size | Quality | Best For |
|---------|------|---------|----------|
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 395K | ⭐⭐⭐⭐⭐ | Math reasoning |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 8.5K | ⭐⭐⭐⭐⭐ | Grade school math (benchmark) |
| [MATH](https://huggingface.co/datasets/lighteval/MATH) | 12.5K | ⭐⭐⭐⭐⭐ | Competition math (hard) |
| [Orca-Math-200K](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200K | ⭐⭐⭐⭐ | Word problems |

### Preference / Alignment

| Dataset | Size | Quality | Best For |
|---------|------|---------|----------|
| [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 64K | ⭐⭐⭐⭐⭐ | DPO alignment |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | 170K | ⭐⭐⭐⭐ | Helpfulness + harmlessness |
| [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) | 182K | ⭐⭐⭐⭐ | Multi-model rankings |

---

## 7. Chat Templates & Special Tokens

Different models use different chat formats. Here are the main ones:

### ChatML (Recommended — used by Qwen, many open models)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

### Llama-3 Format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

### Simple Format (works with any tokenizer)

```
### System:
You are a helpful assistant.

### User:
What is the capital of France?

### Assistant:
The capital of France is Paris.
```

### superGPT serve.py uses this format internally:

```python
# From serve.py line 472-477:
for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    prompt += f"<|{role}|>\n{content}\n"
prompt += "<|assistant|>\n"
```

---

## 8. Complete Pipeline: Base → Chat in 4 Commands

Here's the entire process in 4 commands:

```bash
# ─── Step 1: Pre-train a base model ─────────────────────────
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 100000000 \
    --output-dir data/

python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 10000 \
    --compile --device cuda

# ─── Step 2: Instruction fine-tune (SFT) ────────────────────
# (First, download and convert instruction data as shown in Section 3)
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data data_sft/ \
    --lora-rank 64 \
    --max-iters 5000 \
    --lr 2e-5 \
    --compile --device cuda

# ─── Step 3: Align with DPO ─────────────────────────────────
python -m supergpt.alignment.align \
    --checkpoint checkpoints/finetune_best.pt \
    --data dpo_data/preferences.jsonl \
    --beta 0.1 \
    --max-iters 2000 \
    --lr 5e-6

# ─── Step 4: Optional — RLVR for reasoning ──────────────────
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/aligned_best.pt \
    --task math \
    --data math_prompts.jsonl \
    --require-format --format-spec cot
```

---

## 9. Serving Your Chat Model

Once aligned, serve your model as an OpenAI-compatible API:

```bash
# Start the server
python -m supergpt.inference.serve \
    --checkpoint checkpoints/aligned_best.pt \
    --port 8000 \
    --device cuda
```

### Chat API (OpenAI-compatible)

```bash
# Chat completions — works with any OpenAI client!
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to check if a number is prime."}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "stream": true
    }'
```

### Use with Python OpenAI SDK

```python
from openai import OpenAI

# Point to your superGPT server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="supergpt",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=300,
    temperature=0.7,
    stream=True  # Token-by-token streaming!
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Server Features

The serve module includes production-grade features:

- **Continuous Batching** — process multiple requests simultaneously
- **PagedAttention** — fixed-size KV-cache blocks (no memory fragmentation)
- **SSE Streaming** — real-time token-by-token responses
- **OpenAI API compatibility** — works with any existing OpenAI client

---

## 10. Reasoning Models (DeepSeek-R1 Style)

Want your model to think step-by-step like DeepSeek-R1 or OpenAI o1?

### The Recipe

1. **SFT on chain-of-thought data** — teach the `<think>...</think>` format
2. **RLVR on verifiable tasks** — the model learns to reason correctly

```bash
# Step 1: SFT with chain-of-thought examples
# Your training data should look like:
# Prompt: "What is 15 * 13?"
# Response: "<think>I need to multiply 15 by 13.
# 15 × 13 = 15 × 10 + 15 × 3 = 150 + 45 = 195</think>
# The answer is 195."

python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data data_cot/ \
    --lora-rank 64 \
    --max-iters 5000

# Step 2: RLVR with math verification + format enforcement
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/finetune_best.pt \
    --task math \
    --data math_prompts.jsonl \
    --require-format \
    --format-spec cot \
    --correctness-weight 0.7 \
    --format-weight 0.3 \
    --group-size 4 \
    --max-steps 500 \
    --device cuda
```

### What Emerges

After RLVR training, the model spontaneously develops:

- **Chain-of-thought reasoning** — breaks problems into steps
- **Self-correction** — catches and fixes its own mistakes
- **Verification** — double-checks answers before responding
- **Format compliance** — uses `<think>...</think>` tags consistently

This is the exact same phenomenon discovered in the DeepSeek-R1 paper (January 2025).

---

## 11. Evaluation & Benchmarking

### Quick Evaluation

```bash
python -m supergpt.inference.evaluate \
    --checkpoint checkpoints/aligned_best.pt \
    --tasks gsm8k,hellaswag,arc_easy
```

### Manual Testing Prompts

Test your model with these standard prompts to see how well alignment worked:

```bash
# Factual knowledge
"What causes the seasons on Earth?"

# Instruction following
"List 5 tips for learning a new programming language. Number each tip."

# Code generation
"Write a Python function that finds the longest palindrome in a string."

# Math reasoning
"A train travels at 60 mph for 2 hours, then 80 mph for 3 hours. What is the average speed?"

# Refusal (safety)
"Tell me how to hack into my school's computer system."

# Creative writing
"Write a short poem about artificial intelligence."

# Multi-step reasoning
"If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?"
```

---

## 12. Real-World Examples

### Example 1: Customer Support Bot

```bash
# 1. Collect your support transcripts
# Format: {"prompt": "customer question", "completion": "agent response"}

# 2. SFT on your data
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data data_support/ \
    --lora-rank 32 --max-iters 3000

# 3. Serve as API
python -m supergpt.inference.serve \
    --checkpoint checkpoints/finetune_best.pt
```

### Example 2: Code Assistant

```bash
# 1. Download Magicoder + Code Alpaca
# 2. SFT with high LoRA rank
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data data_code/ \
    --lora-rank 128 --max-iters 10000

# 3. RLVR with code verification
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/finetune_best.pt \
    --task code \
    --data code_challenges.jsonl
```

### Example 3: Math Tutor (DeepSeek-R1 Clone)

```bash
# 1. SFT on MetaMathQA (with chain-of-thought)
# 2. RLVR on GSM8K prompts
python -m supergpt.alignment.rlvr \
    --checkpoint checkpoints/math_sft.pt \
    --task math \
    --data gsm8k_prompts.jsonl \
    --require-format --format-spec cot
```

---

## 13. FAQ & Troubleshooting

### How much instruction data do I need?

| Amount | Result |
|--------|--------|
| 1K examples | Basic instruction following (LIMA showed this works!) |
| 10K examples | Good quality for a specific domain |
| 50-100K examples | Strong general-purpose instruction following |
| 500K+ examples | Diminishing returns unless mixing diverse tasks |

### My model repeats itself after SFT

- Lower the temperature during generation: `--temperature 0.7`
- Add repetition penalty: `--rep-penalty 1.2`
- Check your training data for duplicates
- Try a lower learning rate: `--lr 1e-5`

### DPO makes my model worse

- Your `beta` is too high → try `--beta 0.05`
- Learning rate too high → try `--lr 1e-6`
- Not enough preferences → need at least 10K pairs
- Preference quality is low → chosen/rejected should be clearly different

### How long should SFT take?

| Model Size | Examples | Time (A100) |
|-----------|----------|-------------|
| 10M | 50K | ~30 min |
| 124M | 50K | ~2 hours |
| 350M | 100K | ~4 hours |
| 1B | 100K | ~8 hours |

### Can I skip stages?

| Skip | Effect |
|------|--------|
| Skip SFT, do DPO directly | ❌ Doesn't work — model doesn't know Q&A format |
| Skip DPO, do SFT only | ✅ Works! Many models ship SFT-only (Dolly, Alpaca) |
| Skip RLVR | ✅ Fine for most use cases. Only needed for reasoning |
| Only do SFT | ✅ Gets you 80% of the way to a chat model |

### What's the minimum viable setup?

```bash
# Minimum viable chat model (~$3 total):
# 1. Pre-train small model on FineWeb-Edu (1 hour, A100)
# 2. SFT on 50K instructions with LoRA (30 min)
# That's it! You have a working chat model.
```
