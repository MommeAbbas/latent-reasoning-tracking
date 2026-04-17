"""
Runs DeepSeek-R1-Distill-Qwen-1.5B on 500 problems from the MATH benchmark,
extracts per-step logit features, and saves traces to src/data/.

Outputs (saved under src/data/):
  math500_traces.npy        float32  (N, MAX_T, 3)
  math500_labels.npy        int8     (N,)
  math500_trace_lengths.npy int16    (N,)

Usage:
  python -m src.data.math500_llm_runner [--model MODEL] [--n_problems N]
"""

import argparse
import os
import re
import numpy as np

from datasets import load_dataset
from src.data.llm_feature_extractor import FeatureNormalizer
from src.data.gsm8k_llm_runner import (
    _token_entropy,
    _step_perplexity,
    _split_into_steps,
)

DEFAULT_MODEL      = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_T              = 40
MIN_T              = 2
OUT_DIR            = os.path.dirname(__file__)
N_PROBLEMS_DEFAULT = 500


def _extract_boxed(text: str):
    """Extract the content of the last \\boxed{} in text."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def _parse_answer(raw: str):
    """Try to parse a MATH answer as float; fall back to normalised string."""
    if raw is None:
        return None
    cleaned = raw.replace(",", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return cleaned.lower()


def _is_correct(gold, pred, tol: float = 1e-3) -> int:
    if gold is None or pred is None:
        return 0
    if isinstance(gold, float) and isinstance(pred, float):
        return int(abs(gold - pred) <= tol)
    return int(str(gold).strip() == str(pred).strip())


def _extract_model_answer(text: str):
    raw = _extract_boxed(text)
    if raw is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        raw = nums[-1] if nums else None
    return _parse_answer(raw)


def run(model_name: str = DEFAULT_MODEL, n_problems: int = N_PROBLEMS_DEFAULT):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("pip install transformers torch accelerate")

    print(f"[math500_llm_runner] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("[math500_llm_runner] Loading MATH benchmark ...")
    dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

    prompt_template = (
        "<|im_start|>user\n"
        "Solve the following math problem. Show your reasoning step by step. "
        "Put your final answer in \\boxed{{}}.\n\n"
        "{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    all_features, all_labels, all_lengths = [], [], []

    for idx in range(min(n_problems, len(dataset))):
        item   = dataset[idx]
        gold   = _parse_answer(_extract_boxed(item["solution"]))
        prompt = prompt_template.format(question=item["problem"])

        inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        except Exception as e:
            print(f"  [idx={idx}] Generation failed: {e}")
            continue

        generated_ids = out.sequences[0, input_len:]
        full_text     = tokenizer.decode(generated_ids, skip_special_tokens=True)
        steps         = _split_into_steps(full_text)
        T             = max(MIN_T, min(MAX_T, len(steps)))
        steps         = steps[:T]

        scores_list = out.scores
        all_new_ids = out.sequences[0, input_len:].cpu()

        step_token_counts = []
        for step in steps:
            enc = tokenizer(step, add_special_tokens=False)
            step_token_counts.append(max(1, len(enc["input_ids"])))

        total_generated = len(scores_list)
        cum     = 0
        padded  = np.zeros((MAX_T, 3), dtype=np.float32)
        prev_answer = None

        for t, (step, n_tok) in enumerate(zip(steps, step_token_counts)):
            end = min(cum + n_tok, total_generated)
            if cum >= total_generated:
                break

            step_logits = torch.stack(scores_list[cum:end], dim=0).squeeze(1)
            step_ids    = all_new_ids[cum:end]

            entropy    = _token_entropy(step_logits)
            perplexity = _step_perplexity(step_logits, step_ids)

            running_ans = _extract_model_answer(
                tokenizer.decode(all_new_ids[:end], skip_special_tokens=True)
            )
            if prev_answer is None:
                consistency = 0.5
            else:
                consistency = 1.0 if (
                    running_ans is not None and
                    prev_answer is not None and
                    running_ans == prev_answer
                ) else 0.0
            prev_answer = running_ans

            padded[t] = [entropy, consistency, perplexity]
            cum = end

        pred    = _extract_model_answer(full_text)
        correct = _is_correct(gold, pred)

        all_features.append(padded)
        all_labels.append(correct)
        all_lengths.append(T)

        if (idx + 1) % 20 == 0:
            acc = np.mean(all_labels) * 100
            print(f"  [{idx+1}/{n_problems}]  accuracy: {acc:.1f}%")

    if not all_features:
        print("No trajectories collected.")
        return

    n_fit = min(200, len(all_features))
    normalizer = FeatureNormalizer()
    normalizer.fit([f[:all_lengths[i]] for i, f in enumerate(all_features[:n_fit])])

    normed = []
    for i, feat in enumerate(all_features):
        T_i   = all_lengths[i]
        out_f = feat.copy()
        out_f[:T_i] = normalizer.transform(feat[:T_i]).astype(np.float32)
        normed.append(out_f)

    traces  = np.stack(normed, axis=0).astype(np.float32)
    labels  = np.array(all_labels, dtype=np.int8)
    lengths = np.array(all_lengths, dtype=np.int16)

    np.save(os.path.join(OUT_DIR, "math500_traces.npy"),        traces)
    np.save(os.path.join(OUT_DIR, "math500_labels.npy"),        labels)
    np.save(os.path.join(OUT_DIR, "math500_trace_lengths.npy"), lengths)

    print(f"\n[math500_llm_runner] Saved {len(labels)} trajectories.")
    print(f"  Overall accuracy : {labels.mean()*100:.1f}%")
    print(f"  Mean T           : {lengths.mean():.1f} steps")
    print(f"  Files written to : {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS_DEFAULT)
    args = parser.parse_args()
    run(model_name=args.model, n_problems=args.n_problems)
