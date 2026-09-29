"""Every benchmark method reproduces the numpy reference exactly, and the driver's logic holds.

Run from the repository root:  python -m pytest experiments/benchmarks/tests
"""
import time

import numpy as np
import pytest

import methods as bm
import run
from ca_emulators import reference
from ca_emulators.rules import certificate_configurations

# 8 de Bruijn configurations (every cell meets all eight neighbourhoods in the
# first update, so every cell's rule table is exercised in full) plus 4 random ones.
CERTIFIED = bm.Case(N=16, n_rules=3, T=5, S=12)
# Edge case: one rule, a single update, a single sample.
MINIMAL = bm.Case(N=8, n_rules=1, T=2, S=1)


def _certified_x0():
    return np.concatenate([certificate_configurations(CERTIFIED.N), CERTIFIED.data()[2][:4]])


def _assert_exact(name, case, x0=None):
    rules, alloc, x = case.data()
    x = x if x0 is None else x0
    expected = reference.evolve(x, rules, case.n_updates, alloc)
    prepared = bm.METHODS[name].setup(case, rules, alloc, x)
    for _ in range(2):  # the cold call, then a warm one reusing traces and models
        assert np.array_equal(prepared.to_states(prepared.run()), expected), name


@pytest.mark.parametrize("name", list(bm.METHODS))
def test_method_matches_numpy(name):
    _assert_exact(name, CERTIFIED, _certified_x0())
    _assert_exact(name, MINIMAL)


def test_registry():
    assert len(bm.METHODS) == 3 + len(bm.DRIVERS) * len(bm.SELECTORS) - len(bm.UNSUPPORTED)
    assert "xla:lc3" not in bm.METHODS and "while_loop_xla:lc3" not in bm.METHODS
    assert set(bm.DEFAULT_METHODS) <= set(bm.METHODS)
    with pytest.raises(ValueError):
        bm.neural("xla", "lc3")


def test_case_data_is_deterministic_and_valid():
    case = bm.Case(N=64, n_rules=5, T=10, S=7)
    (r1, a1, x1), (r2, a2, x2) = case.data(), case.data()
    assert np.array_equal(r1, r2) and np.array_equal(a1, a2) and np.array_equal(x1, x2)
    assert len(set(r1.tolist())) == 5 and np.all(np.diff(r1) > 0)
    assert a1.shape == (64,) and a1.min() >= 0 and a1.max() < 5
    assert x1.shape == (7, 64) and x1.dtype == np.uint8
    assert not np.array_equal(bm.Case(N=64, n_rules=5, T=10, S=8).data()[2][:7], x1)
    with pytest.raises(ValueError):
        bm.Case(N=64, n_rules=5, T=1, S=7)


def test_selector_bytes():
    case = bm.Case(N=100, n_rules=4, T=3, S=1)
    assert bm.selector_bytes(case, "dense") == bm.selector_bytes(case, "lc2") == 4 * 4 * 100 ** 2
    assert bm.selector_bytes(case, "lc1") == bm.selector_bytes(case, "elem") == 4 * 4 * 100


def test_measure_warm_autorange_and_budget():
    times, number, status, out = run.measure_warm(lambda: 1, repeats=3, min_time=0.002, budget=10)
    assert status == "ok" and len(times) == 3 and number > 1 and out == 1

    def slow():
        time.sleep(0.02)
        return 2

    times, number, status, out = run.measure_warm(slow, repeats=10, min_time=0, budget=0.05)
    assert status == "over_budget" and len(times) == 1 and number == 1


def test_predicted_skip_and_blocking():
    chain = run.Chain()
    assert run.predicted_skip(chain, 64, 10, 1.0, 10.0) == ""
    chain.update({"status": "ok", "value": 32, "warm_median_s": 0.04, "cold_s": 1.0,
                  "vary": "N"})
    assert run.predicted_skip(chain, 64, 10, 1.0, 10.0) == ""        # 0.08 s x 10 < 1 s
    assert "warm" in run.predicted_skip(chain, 128, 10, 1.0, 10.0)   # 0.16 s x 10 > 1 s
    assert "cold" in run.predicted_skip(chain, 512, 10, 100.0, 10.0)  # 16 s > 10 s
    chain.update({"status": "timeout", "value": 64, "vary": "N"})
    assert chain.blocked == "timeout at N=64"


def test_plan_orders_points_and_skips_threads():
    jobs = run.plan(run.SMOKE_SCENARIOS, ["core", "extended"], ["cellpylib", "numpy"],
                    ["default", "single"])
    assert ("cellpylib", "single") not in {(j[4], j[3]) for j in jobs}
    n_values = [j[1] for j in jobs if j[0].name == "N"]
    assert n_values == sorted(n_values)
    assert len(jobs) == 3 * sum(len(s.points()) for s in run.SMOKE_SCENARIOS.values())


def test_worker_in_process():
    spec = dict(method="xla:elem", case=dict(N=16, n_rules=2, T=4, S=3, seed=1), threads="default",
                repeats=2, min_time=0.0, budget=60, cold_budget=60)
    row = run.worker(spec)
    assert row["status"] == "ok" and row["correct"] is True, row
    assert row["repeats"] == 2 and row["warm_q1_s"] <= row["warm_median_s"] <= row["warm_q3_s"]
    assert row["cold_s"] == pytest.approx(row["build_s"] + row["first_call_s"])
