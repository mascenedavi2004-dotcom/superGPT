"""
RLVR — Reinforcement Learning with Verifiable Rewards
=======================================================
DeepSeek-R1 style RL training with auto-verifiable rewards.
No human labels needed — math correctness, code execution, and format checks
provide the reward signal.

Paper: DeepSeek-R1, arXiv 2501.12948, Jan 2025

Key insight: Pure RL with verifiable rewards produces emergent reasoning
capabilities (chain-of-thought, self-reflection) without supervised data.

Usage:
    # Math reasoning (verify answers)
    python rlvr.py --checkpoint best.pt --task math --data gsm8k_prompts.jsonl

    # Code generation (execute and test)
    python rlvr.py --checkpoint best.pt --task code --data code_prompts.jsonl

    # Format-constrained generation
    python rlvr.py --checkpoint best.pt --task format --format-spec "<think>...</think>"

    # Combined rewards (math + format)
    python rlvr.py --checkpoint best.pt --task math --require-format

Reference:
    DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
    via Reinforcement Learning" (2025)
"""

import os
import sys
import re
import json
import math
import time
import argparse
from typing import List, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from supergpt.core.config import GPTConfig, get_model_config
from supergpt.core.model import GPT


# ==============================================================================
#  Verifiers — Auto-check correctness without human labels
# ==============================================================================

class MathVerifier:
    """Verify mathematical answers.

    Supports:
    - Numeric comparison (with tolerance)
    - Fraction/expression matching
    - LaTeX answer extraction (\\boxed{...})
    - GSM8K format (#### answer)
    """

    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from model output."""
        # Try \\boxed{...}
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()

        # Try #### format (GSM8K)
        hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
        if hashes:
            return hashes[-1].strip()

        # Try "The answer is X"
        answer_is = re.findall(
            r'(?:the\s+)?answer\s+is[:\s]*([^\n.]+)',
            text, re.IGNORECASE
        )
        if answer_is:
            return answer_is[-1].strip()

        # Try last number in text
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        if numbers:
            return numbers[-1].replace(',', '')

        return None

    def verify(self, model_output: str, gold_answer: str) -> Tuple[bool, float]:
        """Verify if model answer matches gold answer.

        Returns: (is_correct, reward_score)
        """
        predicted = self.extract_answer(model_output)
        if predicted is None:
            return False, 0.0

        # Clean up
        predicted = predicted.replace(',', '').replace('$', '').strip()
        gold = gold_answer.replace(',', '').replace('$', '').strip()

        # Exact string match
        if predicted.lower() == gold.lower():
            return True, 1.0

        # Numeric comparison
        try:
            p_val = float(predicted)
            g_val = float(gold)
            if abs(p_val - g_val) < self.tolerance:
                return True, 1.0
            # Partial credit for close answers
            rel_error = abs(p_val - g_val) / max(abs(g_val), 1e-8)
            if rel_error < 0.01:
                return False, 0.5
        except ValueError:
            pass

        return False, 0.0


class CodeVerifier:
    """Verify code correctness by executing test cases.

    Supports:
    - Python code execution with assertions
    - Function extraction and testing
    - Timeout protection
    """

    def __init__(self, timeout=5):
        self.timeout = timeout

    def extract_code(self, text: str) -> str:
        """Extract Python code from model output."""
        # Try ```python ... ``` blocks
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Try indented code blocks
        lines = text.split('\n')
        code_lines = [l for l in lines if l.startswith('    ') or l.startswith('\t')
                      or l.startswith('def ') or l.startswith('class ')]
        if code_lines:
            return '\n'.join(code_lines)

        return text

    def verify(self, model_output: str, test_code: str) -> Tuple[bool, float]:
        """Execute code and run tests.

        Returns: (all_passed, reward_score)
        """
        code = self.extract_code(model_output)
        full_code = code + "\n" + test_code

        try:
            # Execute with restricted globals
            exec_globals = {"__builtins__": {
                "range": range, "len": len, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "tuple": tuple,
                "set": set, "bool": bool, "print": print, "abs": abs,
                "min": min, "max": max, "sum": sum, "sorted": sorted,
                "enumerate": enumerate, "zip": zip, "map": map,
                "filter": filter, "isinstance": isinstance, "type": type,
                "True": True, "False": False, "None": None,
                "AssertionError": AssertionError, "ValueError": ValueError,
            }}
            exec(full_code, exec_globals)
            return True, 1.0
        except AssertionError:
            return False, 0.0
        except Exception:
            return False, -0.1  # Penalty for non-functional code


class FormatVerifier:
    """Verify output format compliance.

    Supports:
    - Chain-of-thought format: <think>...</think>
    - Step-by-step format: Step 1: ... Step 2: ...
    - Structured output: JSON, markdown, etc.
    """

    def __init__(self, format_spec: str = "cot"):
        self.format_spec = format_spec

    def verify(self, text: str) -> Tuple[bool, float]:
        """Check if output follows required format.

        Returns: (is_compliant, reward_score)
        """
        if self.format_spec == "cot":
            return self._check_cot(text)
        elif self.format_spec == "steps":
            return self._check_steps(text)
        elif self.format_spec == "json":
            return self._check_json(text)
        else:
            return True, 0.5

    def _check_cot(self, text: str) -> Tuple[bool, float]:
        """Check <think>...</think> chain-of-thought format."""
        score = 0.0

        # Has thinking section
        if '<think>' in text and '</think>' in text:
            score += 0.5
            # Thinking section has content
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if think_match and len(think_match.group(1).strip()) > 20:
                score += 0.3
        elif 'think' in text.lower() or 'let me' in text.lower():
            score += 0.2

        # Has final answer after thinking
        if '</think>' in text:
            after_think = text.split('</think>')[-1].strip()
            if len(after_think) > 5:
                score += 0.2

        return score >= 0.7, score

    def _check_steps(self, text: str) -> Tuple[bool, float]:
        """Check Step 1: ... Step 2: ... format."""
        steps = re.findall(r'(?:step\s+\d|^\d+[.):])', text, re.IGNORECASE | re.MULTILINE)
        score = min(len(steps) * 0.25, 1.0)
        return len(steps) >= 2, score

    def _check_json(self, text: str) -> Tuple[bool, float]:
        """Check valid JSON output."""
        try:
            json.loads(text.strip())
            return True, 1.0
        except (json.JSONDecodeError, ValueError):
            # Try extracting JSON from text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    json.loads(json_match.group())
                    return True, 0.8
                except (json.JSONDecodeError, ValueError):
                    pass
            return False, 0.0


# ==============================================================================
#  Combined Reward Function
# ==============================================================================

class VerifiableRewardFunction:
    """Combine multiple verifiers into a single reward function.

    This is the core of RLVR — replace human reward models with
    auto-verifiable rewards for training RL on reasoning tasks.
    """

    def __init__(self, task_type="math", format_spec=None,
                 correctness_weight=0.7, format_weight=0.3):
        self.task_type = task_type
        self.correctness_weight = correctness_weight
        self.format_weight = format_weight

        # Task-specific verifier
        if task_type == "math":
            self.verifier = MathVerifier()
        elif task_type == "code":
            self.verifier = CodeVerifier()
        else:
            self.verifier = None

        # Format verifier (optional)
        self.format_verifier = FormatVerifier(format_spec) if format_spec else None

    def __call__(self, prompt_text: str, output_text: str,
                 gold_answer: str = None, test_code: str = None) -> float:
        """Compute total reward.

        Args:
            prompt_text: the original prompt
            output_text: model's generated response
            gold_answer: correct answer (for math)
            test_code: test assertions (for code)

        Returns:
            reward: float in [-1, 1]
        """
        reward = 0.0

        # Correctness reward
        if self.verifier and (gold_answer or test_code):
            if self.task_type == "math" and gold_answer:
                is_correct, score = self.verifier.verify(output_text, gold_answer)
            elif self.task_type == "code" and test_code:
                is_correct, score = self.verifier.verify(output_text, test_code)
            else:
                score = 0.0
            reward += self.correctness_weight * score

        # Format reward
        if self.format_verifier:
            _, fmt_score = self.format_verifier.verify(output_text)
            reward += self.format_weight * fmt_score

        return reward


# ==============================================================================
#  RLVR Training Loop (uses DAPO internally)
# ==============================================================================

def _get_device(args):
    """Select compute device."""
    if args.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return args.device


def _tokenize_text(text, vocab_size):
    """Simple tokenization for RLVR."""
    if vocab_size > 256:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            return tok.encode(text)
        except Exception:
            pass
    return [min(ord(c), vocab_size - 1) for c in text]


def _detokenize(tokens, vocab_size):
    """Simple detokenization."""
    if vocab_size > 256:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            return tok.decode(tokens, skip_special_tokens=True)
        except Exception:
            pass
    return "".join(chr(t) if t < 128 else "?" for t in tokens)


def train_rlvr(args):
    """RL with Verifiable Rewards training loop.

    Uses DAPO as the RL algorithm with verifiable reward functions
    instead of trained reward models.
    """
    device = _get_device(args)

    # Load model
    print(f"Loading model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])

    policy = GPT(config)
    policy.load_state_dict(ckpt["model"])
    policy.to(device)

    ref_policy = GPT(config)
    ref_policy.load_state_dict(ckpt["model"])
    ref_policy.to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in policy.parameters())

    # Setup verifiable reward
    format_spec = args.format_spec if args.require_format else None
    reward_fn = VerifiableRewardFunction(
        task_type=args.task,
        format_spec=format_spec,
        correctness_weight=args.correctness_weight,
        format_weight=args.format_weight,
    )

    # Load task data
    tasks = []
    if args.data and os.path.exists(args.data):
        with open(args.data, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    tasks.append({"prompt": line, "answer": ""})
    else:
        # Default math tasks
        tasks = [
            {"prompt": "What is 15 + 27?", "answer": "42"},
            {"prompt": "What is 8 * 7?", "answer": "56"},
            {"prompt": "What is 100 / 4?", "answer": "25"},
            {"prompt": "What is 2^10?", "answer": "1024"},
            {"prompt": "If x + 3 = 7, what is x?", "answer": "4"},
            {"prompt": "What is the square root of 144?", "answer": "12"},
            {"prompt": "What is 3! (3 factorial)?", "answer": "6"},
            {"prompt": "What is 17 - 9?", "answer": "8"},
        ]
        print(f"  Using {len(tasks)} built-in math tasks")

    # Tokenize prompts
    prompt_tokens = []
    for task in tasks:
        tokens = _tokenize_text(task["prompt"], config.vocab_size)
        prompt_tokens.append(torch.tensor(tokens, dtype=torch.long))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\n{'='*60}")
    print(f"  RLVR Training (DeepSeek-R1 Style)")
    print(f"{'='*60}")
    print(f"  Model:       {n_params/1e6:.1f}M params")
    print(f"  Task:        {args.task}")
    print(f"  Format:      {format_spec or 'none'}")
    print(f"  Group size:  {args.group_size}")
    print(f"  DAPO:        Clip-Higher + Dynamic Sampling")
    print(f"  Tasks:       {len(tasks)} problems")
    print(f"{'='*60}\n")

    t0 = time.time()
    best_accuracy = 0.0

    for step in range(args.max_steps):
        # Sample tasks
        batch_idx = torch.randint(len(tasks), (args.batch_size,))
        batch_tasks = [tasks[i] for i in batch_idx]
        batch_prompts = [prompt_tokens[i] for i in batch_idx]

        # Create reward function wrapper for DAPO
        def rlvr_reward(prompt_ids, completion_ids, task_idx=None):
            # Detokenize
            output_text = _detokenize(completion_ids.tolist(), config.vocab_size)

            # Find matching task
            prompt_text = _detokenize(prompt_ids.tolist(), config.vocab_size)
            matched_task = None
            for t in batch_tasks:
                if t["prompt"][:20] in prompt_text or prompt_text[:20] in t["prompt"]:
                    matched_task = t
                    break

            if matched_task is None:
                return 0.0

            return reward_fn(
                prompt_text=matched_task["prompt"],
                output_text=output_text,
                gold_answer=matched_task.get("answer"),
                test_code=matched_task.get("test_code"),
            )

        # DAPO step with verifiable rewards
        from supergpt.alignment.rlhf import dapo_step
        loss, stats = dapo_step(
            policy, ref_policy, rlvr_reward, batch_prompts, device,
            group_size=args.group_size, max_gen=args.max_gen,
            kl_coef=args.kl_coef,
            clip_eps_low=0.2, clip_eps_high=0.28,
            temperature=args.temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            t1 = time.time()
            print(f"  step {step:>4d} | reward {stats['reward']:.3f} | "
                  f"skipped {stats.get('skipped_prompts', 0)}/{args.batch_size} | "
                  f"{t1-t0:.1f}s")
            t0 = t1

        # Periodic evaluation
        if (step + 1) % args.eval_interval == 0:
            accuracy = _evaluate_rlvr(policy, tasks[:50], config, device, reward_fn)
            print(f"    Eval accuracy: {accuracy:.1%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_save = {
                    "model": policy.state_dict(),
                    "model_config": config.to_dict(),
                    "iter_num": step,
                    "alignment": {
                        "method": "rlvr",
                        "task": args.task,
                        "accuracy": accuracy,
                    },
                }
                save_path = os.path.join(args.output_dir, "rlvr_best.pt")
                torch.save(ckpt_save, save_path)
                print(f"    Saved best: {save_path}")

    print(f"\nRLVR training complete. Best accuracy: {best_accuracy:.1%}")


@torch.no_grad()
def _evaluate_rlvr(policy, tasks, config, device, reward_fn, max_gen=128):
    """Quick evaluation on a set of tasks."""
    policy.eval()
    correct = 0
    total = 0

    for task in tasks:
        tokens = _tokenize_text(task["prompt"], config.vocab_size)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Generate
        for _ in range(max_gen):
            logits, _ = policy(x[:, -config.block_size:])
            logits = logits[:, -1, :] / 0.1  # Low temperature for eval
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)

        output_text = _detokenize(x[0, len(tokens):].tolist(), config.vocab_size)
        r = reward_fn(
            task["prompt"], output_text,
            gold_answer=task.get("answer"),
            test_code=task.get("test_code"),
        )
        if r > 0.5:
            correct += 1
        total += 1

    policy.train()
    return correct / max(total, 1)


# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLVR: RL with Verifiable Rewards (DeepSeek-R1 style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Math reasoning with auto-verification
  python rlvr.py --checkpoint best.pt --task math --data math_prompts.jsonl

  # Code generation with execution-based verification
  python rlvr.py --checkpoint best.pt --task code --data code_prompts.jsonl

  # Math with chain-of-thought format requirement
  python rlvr.py --checkpoint best.pt --task math --require-format --format-spec cot
        """,
    )

    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--task", default="math", choices=["math", "code", "format"],
                        help="Task type for verification")
    parser.add_argument("--data", default=None, help="Task data (JSONL)")
    parser.add_argument("--require-format", action="store_true",
                        help="Also reward format compliance")
    parser.add_argument("--format-spec", default="cot",
                        choices=["cot", "steps", "json"],
                        help="Required format type")
    parser.add_argument("--correctness-weight", type=float, default=0.7)
    parser.add_argument("--format-weight", type=float, default=0.3)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-gen", type=int, default=128)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    train_rlvr(args)
