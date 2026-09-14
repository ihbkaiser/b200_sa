import torch

from data.benchmark_metrics import (
    LONG_BENCH_METRICS,
    extract_boxed,
    extract_gpqa_choice,
    score_aime,
    score_gpqa,
    score_longbench_v2,
    score_math500,
)
from data.dataset import Dataset


class _CharacterTokenizer:
    chat_template = None
    bos_token = "B"

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        assert return_tensors == "pt"
        assert not add_special_tokens
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)


def test_aime_boxed_answer():
    assert score_aime(r"work ... \boxed{70}", "70")
    assert not score_aime(r"work ... \boxed{11}", "70")


def test_boxed_parser_balances_nested_latex_and_task_choice():
    prediction = r"first \boxed{\frac{1}{2}} then \boxed{70}"
    assert extract_boxed(prediction) == r"\frac{1}{2}"
    assert score_aime(prediction, "70")
    assert score_math500(prediction, r"\frac{1}{2}")


def test_longbench_v2_matches_current_kvpress_phrase():
    assert score_longbench_v2("The correct answer is **(C)**.", "C")
    assert not score_longbench_v2("I considered choice C.", "C")


def test_longbench_v2_middle_truncates_complete_prompt():
    dataset = object.__new__(Dataset)
    dataset.dataset_name = "longbench-v2"
    dataset.tokenizer = _CharacterTokenizer()
    dataset.datalen = 10

    actual = dataset._tokenize_benchmark_prompt(
        context="abcdefghijklmno", question="QRST", answer_prefix=""
    )

    # Complete prompt is "BabcdefghijklmnoQRST\n".  Official LongBench-v2
    # truncation retains both ends, including the question at the end.
    expected = torch.tensor(
        [[ord(char) for char in "BabcdQRST\n"]], dtype=torch.long
    )
    assert torch.equal(actual, expected)


def test_gpqa_uses_last_deliberate_answer():
    prediction = "Answer: A\nAfter checking, final answer: (C)"
    assert extract_gpqa_choice(prediction) == "C"
    assert score_gpqa(prediction, "C")


def test_longbench_retrieval_matches_official_number_rule():
    metric = LONG_BENCH_METRICS["passage_retrieval_en"]
    assert metric("15.", "Paragraph 15") == 1.0


def test_longbench_qa_f1_is_fractional():
    metric = LONG_BENCH_METRICS["narrativeqa"]
    score = metric(
        "He lives with the Mulvilles.",
        "He is a guest in the home of the Mulvilles.",
    )
    assert 0.0 < score < 1.0
