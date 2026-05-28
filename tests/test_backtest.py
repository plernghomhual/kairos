import pytest
from kairos.backtest.runner import BacktestResult, evaluate_hit_rate


def test_hit_rate_calculation():
    signals = [
        {"direction": "bullish", "triggered_at": "2021-01-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-02-01", "estimated_hours": 36},
        {"direction": "bearish", "triggered_at": "2021-03-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-04-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-05-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-06-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-07-01", "estimated_hours": 36},
        {"direction": "bearish", "triggered_at": "2021-08-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-09-01", "estimated_hours": 36},
        {"direction": "bullish", "triggered_at": "2021-10-01", "estimated_hours": 36},
    ]
    actuals = [True, True, True, True, True, True, True, False, False, False]

    result = evaluate_hit_rate(signals, actuals)
    assert isinstance(result, BacktestResult)
    assert result.total == 10
    assert result.hits == 7
    assert abs(result.hit_rate - 0.7) < 0.001
    assert result.passes_threshold == True


def test_below_threshold():
    signals = [{"direction": "bullish", "triggered_at": "2021-01-01", "estimated_hours": 36}] * 10
    actuals = [True] * 5 + [False] * 5
    result = evaluate_hit_rate(signals, actuals)
    assert result.hit_rate == 0.5
    assert result.passes_threshold == False
