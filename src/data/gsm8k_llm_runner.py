"""
Runs a small open-source LLM on GSM8K problems, extracts per-step features
from model logits, and saves the results to src/data/.

Features (all derived from model internals, not surface text):
  y[t, 0]  token entropy       — mean entropy of next-token distribution per step
                                  high = uncertain/exploring, low = confident
  y[t, 1]  answer consistency  — 1 if running answer matches previous step's answer
                                  captures coherence of the reasoning chain
  y[t, 2]  step perplexity     — normalised mean NLL of tokens in this step
                                  high = model is surprised by its own output (fatigue)

Outputs (saved under src/data/):
  real_traces.npy        float32  (N, MAX_T, 3)
  real_labels.npy        int8     (N,)
  real_trace_lengths.npy int16    (N,)

Usage:
  python -m src.data.gsm8k_llm_runner [--model MODEL] [--n_problems N]
"""

import argparse
import os
import re
import numpy as np

from datasets import load_dataset
from src.data.llm_feature_extractor import FeatureNormalizer

DEFAULT_MODEL      = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_T              = 40
MIN_T              = 2
OUT_DIR            = os.path.dirname(__file__)
N_PROBLEMS_DEFAULT = 300


# ── answer parsing ────────────────────────────────────────────────────────────
def _parse_number(text: str):
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None

def _extract_gold_answer(answer_text: str):
    if "####" in answer_text:
        return _parse_number(answer_text.split("####")[-1].strip())
    return _parse_number(answer_text)

def _extract_model_answer(text: str):
    if "####" in text:
        return _parse_number(text.split("####")[-1].strip())
    return _parse_number(text)

def _is_correct(gold, pred, tol: float = 1e-3) -> int:
    if gold is None or pred is None:
        return 0
    return int(abs(gold - pred) <= tol)

def _split_into_steps(text: str):
    raw = [s.strip() for s in text.split("\n") if s.strip()]
    return raw if raw else [text.strip()]


# ── logit-based feature extraction ───────────────────────────────────────────
def _token_entropy(logits_tensor) -> float:
    """
    Mean per-token entropy of the next-token distribution.
    High entropy = model is uncertain (maps to uncertainty proxy).
    """
    import torch
    import torch.nn.functional as F
    probs = F.softmax(logits_tensor.float(), dim=-1)          # (seq_len, vocab)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)                # (seq_len,)
    # Normalise by log(vocab_size) so range is [0, 1]
    max_entropy = np.log(logits_tensor.shape[-1])
    return float(entropy.mean().item() / max_entropy)


def _step_perplexity(logits_tensor, input_ids_tensor) -> float:
    """
    Mean per-token NLL for the tokens in this step (normalised to [0, 1]).
    High = model is surprised by its own output (fatigue proxy).
    """
    import torch
    import torch.nn.functional as F
    # logits: (seq_len, vocab), input_ids: (seq_len,)
    # shift: predict token t+1 from logits at t
    if logits_tensor.shape[0] < 2:
        return 0.0
    log_probs = F.log_softmax(logits_tensor[:-1].float(), dim=-1)
    targets   = input_ids_tensor[1:].long()
    nll       = -log_probs[range(len(targets)), targets]
    # normalise by log(vocab_size)
    max_nll = np.log(logits_tensor.shape[-1])
    return float(nll.mean().item() / max_nll)


def _extract_step_features(step_outputs, prev_answer) -> tuple:
    """
    Given HuggingFace generate() output with scores, extract 3 features.
    Returns (entropy, consistency, perplexity, running_answer).
    """
    import torch

    # step_outputs.scores: tuple of (vocab_size,) tensors, one per generated token
    # step_outputs.sequences: (1, seq_len) input+output token ids
    scores = step_outputs.get("scores", None)
    sequences = step_outputs.get("sequences", None)

    if scores is None or len(scores) == 0:
        return 0.5, 0.5, 0.5, prev_answer

    # Stack scores: (num_new_tokens, vocab)
    logits = torch.stack(scores, dim=0).squeeze(1)            # (T_gen, vocab)
    # Generated token ids (new tokens only)
    input_len = sequences.shape[1] - len(scores)
    new_ids   = sequences[0, input_len:]                      # (T_gen,)

    entropy     = _token_entropy(logits)
    perplexity  = _step_perplexity(logits, new_ids)

    # Answer consistency: does this step change the running answer?
    step_text   = ""
    return entropy, 0.0, perplexity, step_text   # consistency filled below


# ── main runner ───────────────────────────────────────────────────────────────
def run(model_name: str = DEFAULT_MODEL, n_problems: int = N_PROBLEMS_DEFAULT):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("pip install transformers torch accelerate")

    print(f"[gsm8k_llm_runner] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("[gsm8k_llm_runner] Loading GSM8K ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    prompt_template = (
        "Solve the following math problem step by step.\n\n"
        "Problem: {question}\n\nSolution:"
    )

    all_features, all_labels, all_lengths = [], [], []

    for idx in range(min(n_problems, len(dataset))):
        item     = dataset[idx]
        gold     = _extract_gold_answer(item["answer"])
        prompt   = prompt_template.format(question=item["question"])

        inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,         # ← get token logits
                )
        except Exception as e:
            print(f"  [idx={idx}] Generation failed: {e}")
            continue

        # Decode full output
        generated_ids = out.sequences[0, input_len:]
        full_text     = tokenizer.decode(generated_ids, skip_special_tokens=True)
        steps         = _split_into_steps(full_text)
        T             = max(MIN_T, min(MAX_T, len(steps)))
        steps         = steps[:T]

        # ── per-step logit features ──────────────────────────────────────────
        # Re-tokenize step-by-step to get per-step logits
        # (cheaper approach: slice the full logit sequence by step token counts)
        scores_list = out.scores           # tuple: one tensor per generated token
        all_new_ids = out.sequences[0, input_len:].cpu()

        # Map each generated token to its step
        step_token_counts = []
        for step in steps:
            enc = tokenizer(step, add_special_tokens=False)
            step_token_counts.append(max(1, len(enc["input_ids"])))

        # Clip to actual generated length
        total_generated = len(scores_list)
        cum = 0
        padded = np.zeros((MAX_T, 3), dtype=np.float32)
        prev_answer = None

        for t, (step, n_tok) in enumerate(zip(steps, step_token_counts)):
            end = min(cum + n_tok, total_generated)
            if cum >= total_generated:
                break

            import torch as _torch
            step_logits = _torch.stack(scores_list[cum:end], dim=0).squeeze(1)
            step_ids    = all_new_ids[cum:end]

            entropy    = _token_entropy(step_logits)
            perplexity = _step_perplexity(step_logits, step_ids)

            # Answer consistency: 1 if answer unchanged from previous step
            running_ans = _extract_model_answer(
                tokenizer.decode(all_new_ids[:end], skip_special_tokens=True)
            )
            if prev_answer is None:
                consistency = 0.5
            else:
                consistency = 1.0 if (
                    running_ans is not None and
                    prev_answer is not None and
                    abs(running_ans - prev_answer) < 1e-3
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

    # ── normalise (fit on first 200, apply to all) ────────────────────────────
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

    np.save(os.path.join(OUT_DIR, "real_traces.npy"),        traces)
    np.save(os.path.join(OUT_DIR, "real_labels.npy"),        labels)
    np.save(os.path.join(OUT_DIR, "real_trace_lengths.npy"), lengths)

    print(f"\n[gsm8k_llm_runner] Saved {len(labels)} trajectories.")
    print(f"  Overall accuracy : {labels.mean()*100:.1f}%")
    print(f"  Mean T           : {lengths.mean():.1f} steps")
    print(f"  Files written to : {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS_DEFAULT)
    args = parser.parse_args()
    run(model_name=args.model, n_problems=args.n_problems)
