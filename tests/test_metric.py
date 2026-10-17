import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src" if (_root / "src" / "ber").exists() else _root))  # repository or package layout

from ber.metric import entity_f05, macro_f05, parse_id_list


def test_pdf_example_2tp_1fp():
    assert math.isclose(entity_f05({"a", "b"}, {"a", "b", "c"}), 5 / 7)


def test_singleton_cases():
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05(set(), {"x"}) == 0.0


def test_nonempty_truth_empty_pred():
    assert entity_f05({"a"}, set()) == 0.0


def test_fn_weight_quarter():
    # 1 tp, 0 fp, 1 fn -> 1.25 / (1.25 + 0.25)
    assert math.isclose(entity_f05({"a", "b"}, {"a"}), 1.25 / 1.5)


def test_macro_missing_pred_counts_empty():
    truth = {"S1-1": [], "S1-2": ["S2-1"]}
    assert math.isclose(macro_f05(truth, {}), 0.5)


def test_parse_id_list():
    assert parse_id_list("") == []
    assert parse_id_list("S2-1,S3-2") == ["S2-1", "S3-2"]
