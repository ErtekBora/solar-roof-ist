import pytest

from core.formatting import percent_from_ratio, round_half_up, tr_num


@pytest.mark.parametrize(
    "value, decimals, expected",
    [
        (1738.7, 0, "1.739"),
        (1738.72, 0, "1.739"),
        (1738.5, 0, "1.739"),  # round half up; Python's f-format would give 1.738 (half to even)
        (0.5, 0, "1"),
        (2.5, 0, "3"),
        (27.15, 1, "27,2"),
        (6.45, 1, "6,5"),
        (25500.5, 1, "25.500,5"),
        (1234567, 0, "1.234.567"),
        (0, 0, "0"),
    ],
)
def test_tr_num_rounds_half_up(value, decimals, expected):
    assert tr_num(value, decimals) == expected


def test_round_half_up_types():
    assert round_half_up(1738.5) == 1739 and isinstance(round_half_up(1738.5), int)
    assert round_half_up(27.15, 1) == 27.2 and isinstance(round_half_up(27.15, 1), float)


def test_percent_from_ratio_matches_payload_ratio():
    assert percent_from_ratio(0.2715) == pytest.approx(27.2)  # payload stores 0.272
    assert percent_from_ratio(0.2705) == pytest.approx(27.1)
    assert tr_num(percent_from_ratio(0.2705), 1) == "27,1"
