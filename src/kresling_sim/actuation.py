from __future__ import annotations
from dataclasses import dataclass
import bisect
import math

import numpy as np

from kresling_sim.coils import Coils
from kresling_sim.kresling import Kresling

_CHANNELS = ("i_xf", "i_xb", "i_yl", "i_yr", "i_zt", "i_zb")


@dataclass
class _Segment:
    """one commanded step on a channel, from start_time onward, following
    the first-order step response of the paper's PI-controlled current loop:
    i(t) = target + (start_value-target)*exp(-(t-start_time)/tau),
    tau = rise_time_s / ln(9) (rise_time_s = 10-90% rise time)"""

    start_time: float
    start_value: float
    target: float
    rise_time_s: float

    def value_at(self, t: float) -> float:
        if t <= self.start_time:
            return self.start_value
        tau = self.rise_time_s / math.log(9)
        return self.target + (self.start_value - self.target) * math.exp(-(t - self.start_time) / tau)


class _ChannelHistory:
    """full sequence of commanded segments for one channel, so
    `current_at(t)` works for any t -- including earlier than the most
    recent command, not just "whatever's current right now". commands must
    be non-decreasing in t."""

    def __init__(self, rise_time_s: float):
        self.rise_time_s = rise_time_s
        self._start_times = [0.0]
        self._segments = [_Segment(start_time=0.0, start_value=0.0, target=0.0, rise_time_s=rise_time_s)]

    def command(self, target: float, t: float) -> None:
        start_value = self.current_at(t)
        self._start_times.append(t)
        self._segments.append(_Segment(start_time=t, start_value=start_value, target=target, rise_time_s=self.rise_time_s))

    def current_at(self, t: float) -> float:
        # rightmost segment whose start_time <= t
        idx = bisect.bisect_right(self._start_times, t) - 1
        idx = max(idx, 0)
        return self._segments[idx].value_at(t)


class Actuation:
    """helmholtz coils + kresling magnet, with each of the 6 channels
    modeled as an independent first-order lag instead of switching
    instantaneously. `command()` sets a new target at a given sim time
    (drive -> zero for a sense window -> resume, say); `field_at()` reads
    the combined field at any sim time using each channel's actual lagged
    current, not its target. commands/queries can interleave (live sim) or
    all commands can be issued upfront and sampled later (e.g. for
    plotting) -- same answer either way, since each channel keeps its full
    history, not just the latest segment.
    """

    def __init__(self, magnet: Kresling, rise_time_s: float | dict[str, float] = 5e-3):
        """rise_time_s: one value for all 6 channels, or a dict keyed by
        channel name (i_xf, i_xb, i_yl, i_yr, i_zt, i_zb) for per-channel
        rise times -- each channel relaxes independently, so axes (or even
        front/back of the same axis) can lag by different amounts.

        default 5ms is this coil rig's own closed-loop controller design
        target, "we use a t_r value of 5 ms" (Sci Adv 2026 SI, eq. S19,
        b=ln(9)/t_r) -- not to be confused with the IEEE APEC 2025 paper's
        1.5ms *measured* step response (Fig. 6, a different quantity: an
        observed result under different tuning/conditions, not this
        platform's designed bandwidth)."""
        self.magnet = magnet
        if isinstance(rise_time_s, dict):
            tr = {ch: rise_time_s.get(ch, 5e-3) for ch in _CHANNELS}
        else:
            tr = {ch: rise_time_s for ch in _CHANNELS}
        self._channels = {ch: _ChannelHistory(rise_time_s=tr[ch]) for ch in _CHANNELS}

    def command(self, currents: dict[str, float], t: float) -> None:
        """set a new target current for any subset of channels, starting at
        sim time t. each named channel relaxes from its actual current at t
        (not its old target); unnamed channels keep their existing target.
        t must be non-decreasing across calls."""
        for ch, target in currents.items():
            self._channels[ch].command(target, t)

    def currents_at(self, t: float) -> dict[str, float]:
        """each channel's actual (lagged) current at sim time t."""
        return {ch: history.current_at(t) for ch, history in self._channels.items()}

    def field_at(self, points, t: float):
        """combined coil + magnet field (mT) at point(s) and sim time t,
        using each channel's actual current then -- so sampling right after
        command() still sees the transient, not the target instantly."""
        coils = Coils(**self.currents_at(t))
        return np.asarray(coils.getB(points)) + np.asarray(self.magnet.getB(points))
