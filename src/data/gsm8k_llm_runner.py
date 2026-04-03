"""
Runs a small open-source LLM on GSM8K problems, extracts per-step features,
and saves the results to src/data/.

Outputs (all saved under src/data/):
  real_traces.npy        float32  (N, MAX_T, 3)   — normalised features
  real_labels.npy        int8     (N,)             — 1 correct / 0 incorrect
  real_trace_lengths.npy int16    (N,)             — actual T per problem

Usage:
  python -m src.data.gsm8k_llm_runner [--model MODEL] [--n_problems N]

Model options (all run on CPU, no GPU required):
  Qwen/Qwen2.5-Math-1.5B-Instruct   (default, ~3 GB RAM)
  microsoft/Phi-3-mini-4k-instruct   (fallback, ~7 GB RAM)
"""

import argparse
import os
import re
import numpy as np

from datasets import load_dataset

from src.data.llm_feature_extractor import StepFeatureExtractor, FeatureNormalizer

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_T          = 40
MIN_T          = 2
OUT_DIR        = os.path.join(os.path.dirname(__file__))   # src/data/
N_PROBLEMS_DEFAULT = 300


# ── answer parsing ────────────────────────────────────────────────────────────
def _parse_number(text: str):
    """Extract the last number from a string.  Returns None if not found."""
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def _extract_gold_answer(answer_text: str):
    """Parse the canonical answer after '####' in GSM8K."""
    if "####" in answer_text:
        tail = answer_text.split("####")[-1].strip()
        return _parse_number(tail)
    return _parse_number(answer_text)


def _extract_model_answer(output_text: str):
    """
    Try to find the model's final numeric answer.
    Looks for '#### X' first (matching GSM8K format), then the last number.
    """
    if "####" in output_text:
        tail = output_text.split("####")[-1].strip()
        return _parse_number(tail)
    # fallback: last number in the output
    return _parse_number(output_text)


def _is_correct(gold, pred, tol: float = 1e-3) -> int:
    if gold is None or pred is None:
        return 0
    return int(abs(gold - pred) <= tol)


# ── step parsing ──────────────────────────────────────────────────────────────
def _split_into_steps(text: str):
    """Split model output into reasoning steps by newline boundaries."""
    raw = [s.strip() for s in text.split("\n") if s.strip()]
    return raw if raw else [text.strip()]


# ── main runner ───────────────────────────────────────────────────────────────
def run(model_name: str = DEFAULT_MODEL, n_problems: int = N_PROBLEMS_DEFAULT):
    # ── load model ────────────────────────────────────────────────────────────
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Install transformers to run the LLM: pip install transformers torch"
        )

    print(f"[gsm8k_llm_runner] Loading model: {model_name}")
    generator = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=512,
        do_sample=False,      # greedy decoding for reproducibility
        device_map="auto",    # uses GPU if available, falls back to CPU
    )

    # ── load dataset ──────────────────────────────────────────────────────────
    print("[gsm8k_llm_runner] Loading GSM8K test split ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    extractor = StepFeatureExtractor()

    all_features = []   # list of (T_i, 3) arrays
    all_labels   = []
    all_lengths  = []

    prompt_template = (
        "Solve the following math problem step by step.\n\n"
        "Problem: {question}\n\n"
        "Solution:"
    )

    for idx in range(min(n_problems, len(dataset))):
        item     = dataset[idx]
        question = item["question"]
        gold     = _extract_gold_answer(item["answer"])

        prompt = prompt_template.format(question=question)

        try:
            output = generator(prompt)[0]["generated_text"]
            # strip the prompt from the output
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
        except Exception as e:
            print(f"  [idx={idx}] Generation failed: {e} — skipping.")
            continue

        steps = _split_into_steps(output)
        T = max(MIN_T, min(MAX_T, len(steps)))
        steps = steps[:T]

        features = extractor.extract(steps)    # (T, 3)
        # zero-pad to MAX_T
        padded = np.zeros((MAX_T, 3), dtype=np.float32)
        padded[:len(features)] = features

        pred    = _extract_model_answer(output)
        correct = _is_correct(gold, pred)

        all_features.append(padded)
        all_labels.append(correct)
        all_lengths.append(T)

        if (idx + 1) % 20 == 0:
            acc_so_far = np.mean(all_labels) * 100
            print(f"  [{idx+1}/{n_problems}] accuracy so far: {acc_so_far:.1f}%")

    if not all_features:
        print("No trajectories collected — check model/dataset setup.")
        return

    # ── normalise ─────────────────────────────────────────────────────────────
    # Fit on first 200 problems (or all if fewer), apply to all
    n_fit = min(200, len(all_features))
    normalizer = FeatureNormalizer()
    normalizer.fit([f[:all_lengths[i]] for i, f in enumerate(all_features[:n_fit])])

    normed_features = []
    for i, feat in enumerate(all_features):
        T_i = all_lengths[i]
        normed = feat.copy()
        normed[:T_i] = normalizer.transform(feat[:T_i]).astype(np.float32)
        normed_features.append(normed)

    traces  = np.stack(normed_features, axis=0)          # (N, MAX_T, 3)
    labels  = np.array(all_labels, dtype=np.int8)        # (N,)
    lengths = np.array(all_lengths, dtype=np.int16)      # (N,)

    # ── save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "real_traces.npy"),        traces)
    np.save(os.path.join(OUT_DIR, "real_labels.npy"),        labels)
    np.save(os.path.join(OUT_DIR, "real_trace_lengths.npy"), lengths)

    print(f"\n[gsm8k_llm_runner] Saved {len(labels)} trajectories.")
    print(f"  Overall accuracy: {labels.mean()*100:.1f}%")
    print(f"  Mean T: {lengths.mean():.1f} steps")
    print(f"  Files written to: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS_DEFAULT)
    args = parser.parse_args()
    run(model_name=args.model, n_problems=args.n_problems)
