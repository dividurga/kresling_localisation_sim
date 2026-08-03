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
    """One commanded step on a single channel, from start_time onward,
    following a first-order step response matching the PI-controlled
    current loop in the IEEE paper:
    i(t) = target + (start_value - target) * exp(-(t-start_time)/tau),
    tau = rise_time_s / ln(9) (rise_time_s is the 10-90% rise time)."""

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
    """The full sequence of commanded segments for one channel, so
    `current_at(t)` gives the right answer for ANY t -- including a t
    earlier than the most recent command -- not just "whatever is current
    right now". Commands must be issued in non-decreasing time order."""

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
    """Helmholtz Coils + Kresling magnet, with each of the 6 coil channels
    modeled as an independent first-order lag rather than switching
    instantaneously. Command a new current set at a given sim time via
    `command()` (e.g. drive the coils, then later command all-zero to open a
    sensing window, then command the prior drive to resume actuation), and
    read the combined field at any sim time via `field_at()` -- the coils'
    actual (lagged) currents at that time are used, not the commanded target.
    Commands and queries can be interleaved (a live, forward-stepping sim)
    or all commands can be issued up front and then sampled at arbitrary
    times (e.g. for plotting a whole schedule) -- both give the same answer,
    since each channel keeps its full command history rather than just its
    most recent segment.
    """

    def __init__(self, magnet: Kresling, rise_time_s: float | dict[str, float] = 1.5e-3):
        """rise_time_s: single value applied to all 6 channels, or a dict
        keyed by channel name (i_xf, i_xb, i_yl, i_yr, i_zt, i_zb) for
        per-channel rise times. Each channel relaxes independently, so
        different axes (or even front/back of the same axis) can lag by
        different amounts if given different rise times."""
        self.magnet = magnet
        if isinstance(rise_time_s, dict):
            tr = {ch: rise_time_s.get(ch, 1.5e-3) for ch in _CHANNELS}
        else:
            tr = {ch: rise_time_s for ch in _CHANNELS}
        self._channels = {ch: _ChannelHistory(rise_time_s=tr[ch]) for ch in _CHANNELS}

    def command(self, currents: dict[str, float], t: float) -> None:
        """Set a new target current for any subset of the 6 channels,
        effective starting at sim time t. Each named channel begins
        relaxing from its actual current at t (not its old target) toward
        the new target; channels not named keep their existing target.
        Must be called with non-decreasing t across successive calls."""
        for ch, target in currents.items():
            self._channels[ch].command(target, t)

    def currents_at(self, t: float) -> dict[str, float]:
        """Each channel's actual (lagged) current at sim time t."""
        return {ch: history.current_at(t) for ch, history in self._channels.items()}

    def field_at(self, points, t: float):
        """Combined coil + magnet field (mT) at the given point(s) and sim
        time t, using each channel's actual current at that time -- so a
        point sampled shortly after a command() call will still see the
        transient, not the new target instantaneously."""
        coils = Coils(**self.currents_at(t))
        return np.asarray(coils.getB(points)) + np.asarray(self.magnet.getB(points))
