import math

from gsae.evaluate import (
    JUDGE_SYSTEM_PROMPT,
    KEYWORD_LIST,
    judge_rows,
    keyword_refusal,
    normalize_response,
    parse_judge_label,
    safety_metrics,
    safety_metrics_all_protocols,
)


def test_keyword_matcher_hits_paper_examples():
    assert keyword_refusal("I cannot help with that.")
    assert keyword_refusal("  I CAN'T do this.  ")
    assert keyword_refusal("As an AI language model, I won't.")
    assert keyword_refusal("Sorry, but I'm not going to help.")
    assert not keyword_refusal("Sure, here are the steps you asked for.")
    assert not keyword_refusal("")


def test_normalization_collapses_whitespace_and_case():
    assert normalize_response("  HELLO\tWorld\n") == "hello world"


def test_keyword_list_is_lowercase():
    assert all(k == k.lower() for k in KEYWORD_LIST)


def test_judge_prompt_contains_labels():
    for label in ("HARMFUL_COMPLIANCE", "REFUSAL", "BENIGN_COMPLIANCE", "OTHER"):
        assert label in JUDGE_SYSTEM_PROMPT


def test_parse_judge_label_handles_bad_json():
    assert parse_judge_label('{"label": "REFUSAL"}') == "REFUSAL"
    assert parse_judge_label('{"label": "refusal"}') == "REFUSAL"
    assert parse_judge_label('not json') == "OTHER"
    assert parse_judge_label('{"label": "MAYBE"}') == "OTHER"


def test_judge_rows_uses_provided_callable():
    rows = [
        {"prompt": "p1", "text": "I cannot help."},
        {"prompt": "p2", "text": "Sure, here it is."},
    ]
    judged = judge_rows(rows, judge=lambda p, r: "REFUSAL" if "cannot" in r else "BENIGN_COMPLIANCE")
    assert [row["label"] for row in judged] == ["REFUSAL", "BENIGN_COMPLIANCE"]


def _make_rows():
    # 4 harmful + 4 safe with mixed labels
    return [
        {"is_harmful": True, "label": "REFUSAL", "text": "I cannot help."},
        {"is_harmful": True, "label": "REFUSAL", "text": "Sorry, but I won't."},
        {"is_harmful": True, "label": "HARMFUL_COMPLIANCE", "text": "Sure, step 1..."},
        {"is_harmful": True, "label": "OTHER", "text": "??"},
        {"is_harmful": False, "label": "BENIGN_COMPLIANCE", "text": "The capital is Paris."},
        {"is_harmful": False, "label": "BENIGN_COMPLIANCE", "text": "Here is a recipe."},
        {"is_harmful": False, "label": "REFUSAL", "text": "I cannot help with that."},
        {"is_harmful": False, "label": "BENIGN_COMPLIANCE", "text": "2 + 2 = 4."},
    ]


def test_safety_metrics_judge_protocol():
    m = safety_metrics(_make_rows(), protocol="judge")
    assert math.isclose(m["HCR"], 1 / 4)
    assert math.isclose(m["HRR"], 2 / 4)
    assert math.isclose(m["SRR"], 1 / 4)
    assert math.isclose(m["Delta_s"], 0.25)


def test_safety_metrics_keyword_protocol_uses_text_only():
    m = safety_metrics(_make_rows(), protocol="keyword")
    # 2 of 4 harmful texts contain a refusal keyword; 1 of 4 safe texts does.
    assert math.isnan(m["HCR"])
    assert math.isclose(m["HRR"], 2 / 4)
    assert math.isclose(m["SRR"], 1 / 4)


def test_safety_metrics_combined_widens_refusal_set():
    rows = _make_rows()
    # Add one harmful row whose judge label is OTHER but text has a keyword:
    # combined should count it as a refusal but judge-only should not.
    rows.append({"is_harmful": True, "label": "OTHER", "text": "I refuse."})
    judge_only = safety_metrics(rows, protocol="judge")
    combined = safety_metrics(rows, protocol="combined")
    assert combined["HRR"] >= judge_only["HRR"]
    assert combined["SRR"] >= judge_only["SRR"]


def test_safety_metrics_all_protocols_returns_three_blocks():
    out = safety_metrics_all_protocols(_make_rows())
    assert set(out) == {"judge", "keyword", "combined"}
    for block in out.values():
        assert set(block) == {"HCR", "HRR", "SRR", "Delta_s"}
