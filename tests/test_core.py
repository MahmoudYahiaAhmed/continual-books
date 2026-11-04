import math

import pytest

from cl_books.books import chunk_paragraphs, strip_boilerplate
from cl_books.judge import contains_match, parse_verdict
from cl_books.qa import dedupe, parse_qa_json, split_facts
from cl_books.report import paired_gain_ci, retention


def test_strip_boilerplate_keeps_only_body():
    text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence"
    assert strip_boilerplate(text) == "body text"


def test_strip_boilerplate_rejects_missing_markers():
    with pytest.raises(ValueError):
        strip_boilerplate("no markers here")


def test_chunk_paragraphs_packs_up_to_limit():
    paras = ["word " * 60] * 6  # ~300 chars each
    chunks = chunk_paragraphs("\n\n".join(paras), max_chars=700)
    assert len(chunks) == 3
    assert all(len(c) <= 700 for c in chunks)


def test_parse_qa_json_filters_bad_items():
    raw = 'Sure: [{"question": "Q1?", "answer": "A1", "paraphrase": "P1?"}, {"question": "Q2?"}] done'
    assert parse_qa_json(raw) == [{"question": "Q1?", "answer": "A1", "paraphrase": "P1?"}]
    assert parse_qa_json("not json") == []
    assert parse_qa_json("[broken") == []


def test_dedupe_and_split():
    facts = [{"question": f"Q{i}?", "answer": f"A{i}", "paraphrase": f"P{i}?"} for i in range(10)]
    facts = dedupe(facts + [{"question": "q0?", "answer": "x", "paraphrase": "y"}])
    train, evals = split_facts(facts, n_eval=4, seed=1)
    assert len(train) == 10 and len(evals) == 4
    assert all(e["question"].startswith("P") for e in evals)
    assert split_facts(facts, 4, 1) == (train, evals)


def test_contains_match_and_verdict():
    assert contains_match("The red thumb-mark", "It was a red thumb mark on paper.")
    assert not contains_match("Reuben Hornby", "Walter Hornby")
    assert parse_verdict(" Yes.") and not parse_verdict("no")


def test_retention_counts_only_previously_correct():
    before = [True, True, False, True]
    after = [True, False, True, True]
    assert retention(before, after) == (2, 3)


def test_paired_gain_ci_brackets_gain():
    before = [False] * 50 + [True] * 50
    after = [True] * 80 + [False] * 20
    gain, lo, hi = paired_gain_ci(before, after)
    assert gain == 30 and lo <= gain <= hi and lo > 0


def test_topk_distill_loss_is_minimised_by_teacher():
    torch = pytest.importorskip("torch")
    from cl_books.losses import topk_distill_loss

    teacher_lp = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    ids = torch.tensor([[0, 1, 2]])
    mask = torch.ones(1)
    matching = torch.full((1, 5), -30.0)
    matching[0, :3] = teacher_lp[0]
    wrong = torch.zeros(1, 5)
    wrong[0, 4] = 10.0
    good = topk_distill_loss(matching, ids, teacher_lp, mask).item()
    bad = topk_distill_loss(wrong, ids, teacher_lp, mask).item()
    entropy = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
    assert good == pytest.approx(entropy, abs=1e-4)
    assert bad > good
