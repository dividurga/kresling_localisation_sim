"""Sanity checks for Coils / Actuation. Not a pytest suite (no test
dependency added yet) -- just asserts + printed values to eyeball.

Run with: uv run python tests/sanity_check_coils_actuation.py
"""
from __future__ import annotations

import numpy as np

from kresling_sim.coils import Coils
from kresling_sim.kresling import Kresling
from kresling_sim.actuation import Actuation


def check_coil_matrix_against_paper() -> None:
    """Driving only i_xf=1A should give Bx(0,0,0) ~= 0.326 mT, matching the
    A(0,0,0) matrix in the paper, equation 9."""
    c = Coils(i_xf=1.0)
    bx, by, bz = c.getB(np.asarray((0, 0, 0)))
    print(f"i_xf=1A at origin -> B = ({bx:.4f}, {by:.4f}, {bz:.4f}) mT (expect Bx ~ 0.326 mT)")
    assert abs(bx - 0.326) < 0.01, f"Bx={bx} does not match paper's ~0.326 mT"
    assert abs(by) < 1e-9 and abs(bz) < 1e-9, "off-axis components should be ~0 by symmetry"


def check_full_coil_matrix_against_paper() -> None:
    """Full A(0,0,0) matrix check (Sci Adv 2026 SI, eq. S8; consistent with
    IEEE APEC 2025, eq. 9) -- drive each of the 6 channels alone at 1A and
    check the field at the origin against every entry of the paper's
    matrix, not just i_xf. Y and Z are distinct (0.324/0.321), not both
    the rounded 0.322 previously used here."""
    expected = {
        "i_xf": (0.326, 0.0, 0.0),
        "i_xb": (0.326, 0.0, 0.0),
        "i_yl": (0.0, 0.324, 0.0),
        "i_yr": (0.0, 0.324, 0.0),
        "i_zt": (0.0, 0.0, 0.321),
        "i_zb": (0.0, 0.0, 0.321),
    }
    for channel, exp in expected.items():
        c = Coils(**{channel: 1.0})
        b = c.getB(np.asarray((0, 0, 0)))
        print(f"{channel}=1A -> B = ({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}) mT   (expect {exp})")
        for got, want in zip(b, exp):
            assert abs(got - want) < 0.01, f"{channel}: {b} does not match expected {exp}"


def check_independent_channel_rise_times() -> None:
    """Two channels with different rise_time_s should decay at visibly
    different rates -- confirms per-channel independence, not just a single
    shared tau applied to everything."""
    act = Actuation(
        Kresling(position=np.asarray((0, 0, 0))),
        rise_time_s={"i_xf": 1.5e-3, "i_zt": 15e-3},
    )
    act.command({"i_xf": 50.0, "i_zt": 50.0}, t=0.0)
    act.command({"i_xf": 0.0, "i_zt": 0.0}, t=1.0)

    fast_at_5ms = act.currents_at(1.005)["i_xf"]
    slow_at_5ms = act.currents_at(1.005)["i_zt"]
    print(f"5ms after switch: i_xf (tau~0.68ms) = {fast_at_5ms:.4f} A,  i_zt (tau~6.8ms) = {slow_at_5ms:.4f} A")
    assert fast_at_5ms < slow_at_5ms, "the fast (shorter rise_time_s) channel should have decayed further by the same time"
    assert fast_at_5ms < 0.1, "fast channel should be essentially settled after ~7 of its own time constants"
    assert slow_at_5ms > 10, "slow channel should still be mid-decay after less than 1 of its own time constants"


def check_actuation_transient() -> None:
    """After commanding i_xf: 50A -> 0A at t=0.1s with rise_time_s=1.5ms,
    the current should decay smoothly (not jump instantly) and be
    negligible after several time constants (~5*tau ~ 6.8ms)."""
    act = Actuation(Kresling(position=np.asarray((0, 0, 0))), rise_time_s=1.5e-3)
    act.command({"i_xf": 50.0}, t=0.0)
    print("current right after actuate command, t=0:   ", act.currents_at(0.0)["i_xf"])
    print("current well into actuate phase, t=0.05:     ", act.currents_at(0.05)["i_xf"])

    act.command({"i_xf": 0.0}, t=0.1)
    prev = None
    for dt in [0.0, 0.0005, 0.0015, 0.005, 0.01]:
        t = 0.1 + dt
        i = act.currents_at(t)["i_xf"]
        print(f"  t=0.1+{dt:<7} -> i_xf = {i:.4f} A")
        if prev is not None:
            assert i <= prev + 1e-9, "current should decay monotonically toward the new target"
        prev = i

    assert abs(act.currents_at(0.1)["i_xf"] - 50.0) < 1e-9, "current should not jump instantly at the switch"
    assert act.currents_at(0.11)["i_xf"] < 0.5, "current should be near zero after ~5 tau"


if __name__ == "__main__":
    check_coil_matrix_against_paper()
    check_full_coil_matrix_against_paper()
    check_independent_channel_rise_times()
    check_actuation_transient()
    print("all sanity checks passed")
