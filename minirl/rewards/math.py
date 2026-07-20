"""Math verifier: extraction -> normalization -> comparison.

The pipeline every math-RL reward follows:
  1. EXTRACT the candidate: the model is PROMPTED to end with a marked answer
     (`\\boxed{...}` or "The answer is N") — the format instruction is part of
     the reward spec; failed extraction = reward 0 even for correct math.
  2. NORMALIZE both sides ($, commas, %, latex wrappers, whitespace).
  3. COMPARE. Modern RLVR datasets (GSM8K, DAPO-math-17k, AIME) deliberately
     use numeric-only final answers so this is exact number equality — no
     symbolic math needed (DAPO even converted answers to integers for this).
     Only when a gold label is a symbolic EXPRESSION (MATH's \\frac{\\sqrt2}2)
     do we escalate to the optional `math-verify` dependency (HuggingFace's
     maintained sympy-based grader — the same lineage slime vendors:
     Hendrycks MATH -> deepscaler -> slime/rm_hub/math_utils.py, 493 lines).

Kept deliberately small: this file IS the whole reward for GSM8K-class data.
"""

import re
from fractions import Fraction
from typing import Callable

from minirl.rollout.types import Trajectory

# last number in the text, tolerating commas and decimals: "1,234.5", "-3"
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def extract_boxed(text: str) -> str | None:
    r"""Contents of the LAST \boxed{...}, with real brace matching —
    regex alone breaks on nesting like \boxed{\frac{1}{2}}."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None  # unbalanced braces -> extraction failure -> reward 0


def extract_final_answer(text: str) -> str | None:
    """Candidate answer: last \\boxed{} if present, else the last number."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else None


def _to_number(s: str) -> Fraction | None:
    """Parse '1,234', '$18', '50%', '1/2', '0.5' -> exact Fraction; None if not numeric."""
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace(" ", "")
    percent = s.endswith("%")
    s = s.rstrip("%")
    try:
        value = Fraction(s)  # handles ints, decimals ('0.5'), and fractions ('1/2')
    except (ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def grade_answer(candidate: str | None, gold: str) -> bool:
    """Is the extracted candidate answer equal to the gold label?

    Numeric-vs-numeric compares exactly (Fraction: '0.5' == '1/2' == '50%').
    Non-numeric golds fall back to normalized string equality, then to the
    optional math-verify symbolic check if it is installed.
    """
    if candidate is None:
        return False
    # GSM8K gold labels look like "... #### 42" — take the marked answer.
    gold = gold.split("####")[-1].strip()

    cand_num, gold_num = _to_number(candidate), _to_number(gold)
    if gold_num is not None:
        return cand_num == gold_num  # exact rational equality, no epsilon needed

    norm = lambda s: s.strip().lower().replace(" ", "").replace("\\left", "").replace("\\right", "")
    if norm(candidate) == norm(gold):
        return True

    try:  # symbolic labels (MATH-style): optional dependency, pip install math-verify
        from math_verify import parse, verify

        def _parse(s: str):
            expr = parse(s)
            return expr if expr else parse(f"${s}$")  # bare latex needs $ delimiters

        gold_expr, cand_expr = _parse(gold), _parse(candidate)
        return bool(gold_expr and cand_expr and verify(gold_expr, cand_expr))
    except ImportError:
        return False  # without math-verify, unmatched symbolic labels grade as wrong


def math_reward(response_text: str, gold: str) -> float:
    """The reward: 1.0 iff the response's final answer matches the label."""
    return float(grade_answer(extract_final_answer(response_text), gold))


def make_math_reward_fn(tokenizer, answer_key: str = "answer") -> Callable[[Trajectory], float]:
    """Glue for the collector: decode the RESPONSE tokens only and grade them
    against the label that label-plumbing put in traj.meta (prompt-source meta)."""

    def reward_fn(traj: Trajectory) -> float:
        response = tokenizer.decode(traj.input_ids[traj.prompt_len :], skip_special_tokens=True)
        return math_reward(response, str(traj.meta[answer_key]))

    return reward_fn
