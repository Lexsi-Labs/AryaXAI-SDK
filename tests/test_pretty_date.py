import sys

import pytest

from aryaxai.common.utils import pretty_date
from aryaxai.core.synthetic import SyntheticDataTag


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-01T10:20:30.123456",
        "2026-11-01 10:20:30.123456",
        "2026-11-01T10:20:30",
        "2026-11-01 10:20:30",
        "2026-11-01T10:20:30+00:00",
        pytest.param(
            "2026-11-01T10:20:30Z",
            marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="fromisoformat reads Z from 3.11"),
        ),
    ],
)
def test_pretty_date(value):
    assert pretty_date(value) == "01-11-2026 10:20:30"


def test_pretty_date_without_time():
    assert pretty_date("2026-11-01") == "01-11-2026 00:00:00"


def test_invalid_date_returns_the_input(capsys):
    assert pretty_date("not a date") == "not a date"
    assert "Date format invalid." in capsys.readouterr().out


def test_tag_prints_without_microseconds():
    tag = SyntheticDataTag.model_construct(model_name="m", tag="t", created_at="2026-11-01T10:20:30")
    assert "created_at=01-11-2026 10:20:30" in str(tag)
    assert repr(tag) == str(tag)
