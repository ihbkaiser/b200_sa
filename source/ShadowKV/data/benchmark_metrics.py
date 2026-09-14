"""Self-contained benchmark scorers used by :mod:`data.dataset`.

The LongBench formulas follow the official LongBench scorer used by the
KVPress experiments in this workspace.  AIME-25 and GPQA expose per-example
predicates because ShadowKV's evaluator accumulates scores one generation at
a time.  LongBench-v2, AIME-25 and MATH-500 intentionally mirror
NVIDIA/kvpress@71640b4.  Keeping these small functions here lets the ShadowKV
tree run without importing a sibling checkout.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        excluded = set(string.punctuation)
        return "".join(char for char in value if char not in excluded)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def count_score(prediction, ground_truth, **kwargs):
    del kwargs
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    hits = sum(str(number) == str(ground_truth) for number in numbers)
    return float(hits / len(numbers))


def retrieval_score(prediction, ground_truth, **kwargs):
    del kwargs
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    hits = sum(str(number) == matches[0] for number in numbers)
    return float(hits / len(numbers))


def code_sim_score(prediction, ground_truth, **kwargs):
    del kwargs
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return fuzz.ratio(candidate, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [
        match for match in matches
        if not (match in ground_truth and match != ground_truth)
    ]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def rouge_score(prediction, ground_truth, **kwargs):
    del kwargs
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception as error:
        print(f"An error occurred while computing ROUGE-L: {error}")
        return 0.0


def f1_score(prediction, ground_truth, **kwargs):
    del kwargs
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    del kwargs
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


LONG_BENCH_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def extract_boxed(text, *, last=False):
    """Return one balanced ``boxed{...}`` payload, as in current KVPress."""
    if not isinstance(text, str):
        return None
    marker = "boxed{"
    marker_index = text.rfind(marker) if last else text.find(marker)
    if marker_index == -1:
        return None

    content_start = marker_index + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def score_aime(prediction, answer):
    # AIME models may revise themselves, so current KVPress scores the last box.
    return extract_boxed(prediction, last=True) == str(answer)


def score_math500(prediction, answer):
    # Deliberately match current KVPress: first balanced box, exact text match.
    return extract_boxed(prediction) == str(answer)


def score_longbench_v2(prediction, answer):
    """Official LongBench-v2 exact-choice extraction used by KVPress."""
    if not isinstance(prediction, str):
        return False
    prediction = prediction.replace("*", "")
    answer = str(answer).strip().upper()
    return (
        f"The correct answer is ({answer})" in prediction
        or f"The correct answer is {answer}" in prediction
    )


GPQA_PATTERNS = (
    re.compile(r"\\boxed\{\s*\(?([ABCD])\)?[\s.}]*", re.I),
    re.compile(r"\bfinal answer\b[^ABCD\n]{0,20}\(?([ABCD])\)?\b", re.I),
    re.compile(r"\banswer\b\s*(?:is|:)\s*\**\s*\(?([ABCD])\)?\b", re.I),
    re.compile(r"^\s*\**\(?([ABCD])\)?\**\s*$", re.I | re.M),
)


def extract_gpqa_choice(prediction):
    if not isinstance(prediction, str) or not prediction:
        return None
    for pattern in GPQA_PATTERNS:
        hits = pattern.findall(prediction)
        if hits:
            return hits[-1].upper()
    return None


def score_gpqa(prediction, answer):
    return extract_gpqa_choice(prediction) == str(answer).strip().upper()
