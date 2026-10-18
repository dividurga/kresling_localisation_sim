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
    """one commanded step, first order response
    i(t) = target + (start - target)*exp(-(t - t0)/tau), tau = rise_time/ln9 (10-90 rise)"""

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
    """all segments for one channel so current_at works for any t. commands must be non-decreasing in t"""

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
    """coils + magnet, 6 channels each an independent first order lag.
    command() sets a target at sim time t, field_at() uses the actual lagged currents"""

    def __init__(self, magnet: Kresling, rise_time_s: float | dict[str, float] = 5e-3):
        """rise_time_s: one value for all channels or a dict keyed by channel name.
        default 5ms is the controller design target (sci adv 2026 SI eq S19),
        not the 1.5ms measured step response in the IEEE APEC 2025 paper (fig 6)"""
        self.magnet = magnet
        if isinstance(rise_time_s, dict):
            tr = {ch: rise_time_s.get(ch, 5e-3) for ch in _CHANNELS}
        else:
            tr = {ch: rise_time_s for ch in _CHANNELS}
        self._channels = {ch: _ChannelHistory(rise_time_s=tr[ch]) for ch in _CHANNELS}

    def command(self, currents: dict[str, float], t: float) -> None:
        """new target current for any subset of channels at time t. relaxes from the actual
        current at t, unnamed channels keep their target. t non-decreasing"""
        for ch, target in currents.items():
            self._channels[ch].command(target, t)

    def currents_at(self, t: float) -> dict[str, float]:
        """each channel's actual (lagged) current at sim time t."""
        return {ch: history.current_at(t) for ch, history in self._channels.items()}

    def field_at(self, points, t: float):
        """coil + magnet field (mT) at points and time t, includes the transient"""
        coils = Coils(**self.currents_at(t))
        return np.asarray(coils.getB(points)) + np.asarray(self.magnet.getB(points))
