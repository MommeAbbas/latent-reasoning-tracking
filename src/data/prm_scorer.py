"""
Scores saved GSM8K traces with Qwen2.5-Math-PRM-7B.

Loads real_questions.npy and real_step_texts.npy produced by gsm8k_llm_runner.py,
runs each problem through the PRM, and saves per-problem final-step PRM scores.

Output:
  src/data/real_prm_scores.npy   float32 (N,)  — final-step PRM score per problem

Usage:
  python -m src.data.prm_scorer [--model MODEL]
"""

import argparse
import os
import numpy as np

OUT_DIR       = os.path.dirname(__file__)
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

# Qwen2.5-Math-PRM uses <extra_0> as the step separator token.
# The model outputs a scalar reward per separator; we use the final one.
STEP_SEP = "<extra_0>"


def _build_prm_input(question: str, steps: list) -> str:
    """Format a problem + steps in the Qwen2.5-Math-PRM chat template."""
    system = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    step_text = STEP_SEP.join(steps) + STEP_SEP
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{step_text}<|im_end|>"
    )


def run(model_name: str = DEFAULT_MODEL):
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers torch accelerate")

    q_path = os.path.join(OUT_DIR, "real_questions.npy")
    s_path = os.path.join(OUT_DIR, "real_step_texts.npy")
    for p in (q_path, s_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Re-run `python -m src.data.gsm8k_llm_runner` first."
            )

    questions  = np.load(q_path, allow_pickle=True)
    step_texts = np.load(s_path, allow_pickle=True)
    N = len(questions)
    print(f"[prm_scorer] {N} problems loaded.")

    print(f"[prm_scorer] Loading PRM: {model_name}")
    from transformers import AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    sep_id = tokenizer.convert_tokens_to_ids(STEP_SEP)

    scores = np.zeros(N, dtype=np.float32)

    for i in range(N):
        question = str(questions[i])
        steps    = list(step_texts[i])
        if not steps:
            scores[i] = 0.5
            continue

        text   = _build_prm_input(question, steps)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        input_ids = inputs["input_ids"][0]
        sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) == 0:
            scores[i] = 0.5
            continue

        with torch.no_grad():
            output = model(**inputs)

        # Qwen2.5-Math-PRM returns per-token scores in output.scores or logits
        # Shape: (1, seq_len, 1) or (1, seq_len, 2)
        if hasattr(output, "scores"):
            raw = output.scores  # (1, seq_len, num_labels)
        else:
            raw = output.logits

        raw = raw.squeeze(0)  # (seq_len, num_labels)
        last_pos = sep_positions[-1].item()
        step_scores = raw[last_pos]  # (num_labels,)

        if step_scores.shape[0] == 1:
            # Single scalar — sigmoid to get probability
            score = float(torch.sigmoid(step_scores[0]).item())
        else:
            # Two logits: index 1 = good
            probs = torch.softmax(step_scores.float(), dim=-1)
            score = float(probs[1].item())

        scores[i] = score

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N}]  mean score so far: {scores[:i+1].mean():.3f}")

    out_path = os.path.join(OUT_DIR, "real_prm_scores.npy")
    np.save(out_path, scores)
    print(f"\n[prm_scorer] Saved scores -> {out_path}")
    print(f"  Score range : {scores.min():.3f} – {scores.max():.3f}")
    print(f"  Mean score  : {scores.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(model_name=args.model)
