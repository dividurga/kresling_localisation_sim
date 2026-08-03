"""Interactive sensor-layout designer.

Start from a parametrized rectangular grid (sliders: count/spacing/shared
tilt), then click-drag individual sensors to reposition them and use the
per-sensor sliders to override just that sensor's tilt. Save the resulting
layout to an .npz file that other scripts can load via
kresling_sim.sensors.load_sensors(path).

Run with: uv run python scripts/sensor_gui.py

Controls:
  - Drag a sensor (blue dot) to reposition it.
  - Click a sensor (press+release without moving) to select it (turns red)
    -- the "selected sensor" sliders then edit just that one's tilt.
  - The top sliders + "Regenerate Grid" rebuild the WHOLE layout from
    scratch, discarding individual edits -- use them first, then fine-tune.
  - Type a filename and click "Save" to write positions + orientations.

Caveat: this is a desktop matplotlib widget app -- I (Claude) verified the
grid-generation and save/load logic programmatically, but could not
actually click-drag in this environment, so the interactive mouse handling
below is untested by me beyond code review. Please exercise it yourself
before trusting it.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from kresling_sim.sensors import Sensor, sensor_grid, save_sensors

PICK_RADIUS_PX = 12.0


class SensorGUI:
    def __init__(self):
        self.sensors: list[Sensor] = []
        self.selected: int | None = None
        self.dragging: int | None = None
        self.press_xy: tuple[float, float] | None = None
        self._suppress_selected_callback = False

        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_axes((0.32, 0.32, 0.63, 0.63))
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_title("Sensor layout (top-down view)")
        self.ax.set_aspect("equal")

        self.scatter = self.ax.scatter([], [], s=60, c="tab:blue", zorder=3)
        self.tilt_lines = []  # list of Line2D, one per sensor, showing azimuth direction

        self._build_grid_controls()
        self._build_selected_controls()
        self._build_save_controls()

        self.regenerate(None)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    # ---- widget layout ----

    def _build_grid_controls(self) -> None:
        labels_defaults = [
            ("num_across", 4, 1, 12),
            ("num_below", 4, 1, 12),
            ("sep_horiz (mm)", 10.0, 1.0, 50.0),
            ("sep_vert (mm)", 10.0, 1.0, 50.0),
            ("shared polar (deg)", 0.0, 0.0, 89.0),
            ("shared azimuth (deg)", 0.0, 0.0, 359.0),
        ]
        self.grid_sliders = {}
        for i, (label, default, lo, hi) in enumerate(labels_defaults):
            ax_s = self.fig.add_axes((0.05, 0.92 - i * 0.045, 0.2, 0.03))
            self.grid_sliders[label] = Slider(ax_s, label, lo, hi, valinit=default)

        ax_btn = self.fig.add_axes((0.05, 0.60, 0.2, 0.05))
        self.regen_button = Button(ax_btn, "Regenerate Grid")
        self.regen_button.on_clicked(self.regenerate)

    def _build_selected_controls(self) -> None:
        self.fig.text(0.05, 0.53, "Selected sensor:", fontsize=10, weight="bold")
        ax_p = self.fig.add_axes((0.05, 0.47, 0.2, 0.03))
        ax_a = self.fig.add_axes((0.05, 0.42, 0.2, 0.03))
        self.selected_polar_slider = Slider(ax_p, "polar (deg)", 0.0, 89.0, valinit=0.0)
        self.selected_azimuth_slider = Slider(ax_a, "azimuth (deg)", 0.0, 359.0, valinit=0.0)
        self.selected_polar_slider.on_changed(self.on_selected_angle_change)
        self.selected_azimuth_slider.on_changed(self.on_selected_angle_change)

    def _build_save_controls(self) -> None:
        ax_text = self.fig.add_axes((0.05, 0.30, 0.2, 0.05))
        self.filename_box = TextBox(ax_text, "", initial="sensor_layout.npz")
        ax_save = self.fig.add_axes((0.05, 0.23, 0.2, 0.05))
        self.save_button = Button(ax_save, "Save layout")
        self.save_button.on_clicked(self.on_save)
        self.status_text = self.fig.text(0.05, 0.18, "", fontsize=9, color="tab:green")

    # ---- core actions ----

    def regenerate(self, _event) -> None:
        s = self.grid_sliders
        self.sensors = sensor_grid(
            num_across=int(round(s["num_across"].val)),
            num_below=int(round(s["num_below"].val)),
            sep_horizontal=s["sep_horiz (mm)"].val,
            sep_vertical=s["sep_vert (mm)"].val,
            polar=np.radians(s["shared polar (deg)"].val),
            azimuth=np.radians(s["shared azimuth (deg)"].val),
        )
        self.selected = None
        self.redraw()

    def redraw(self) -> None:
        positions = np.array([sn.position for sn in self.sensors])
        colors = ["tab:red" if i == self.selected else "tab:blue" for i in range(len(self.sensors))]
        self.scatter.set_offsets(positions[:, :2] if len(positions) else np.empty((0, 2)))
        self.scatter.set_color(colors)

        for line in self.tilt_lines:
            line.remove()
        self.tilt_lines = []
        for sn in self.sensors:
            polar, azimuth = _polar_azimuth_from_rotation(sn.orientation)
            length = 3.0 + 6.0 * (polar / (np.pi / 2))  # visually scale with tilt magnitude
            dx, dy = length * np.cos(azimuth), length * np.sin(azimuth)
            (line,) = self.ax.plot(
                [sn.position[0], sn.position[0] + dx],
                [sn.position[1], sn.position[1] + dy],
                color="gray",
                linewidth=1,
                zorder=2,
            )
            self.tilt_lines.append(line)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def _nearest_sensor(self, event) -> int | None:
        if not self.sensors or event.xdata is None:
            return None
        positions = np.array([sn.position[:2] for sn in self.sensors])
        px_positions = self.ax.transData.transform(positions)
        px_mouse = np.array([event.x, event.y])
        dists = np.linalg.norm(px_positions - px_mouse, axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= PICK_RADIUS_PX else None

    def on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        idx = self._nearest_sensor(event)
        if idx is not None:
            self.dragging = idx
            self.press_xy = (event.xdata, event.ydata)

    def on_motion(self, event) -> None:
        if self.dragging is None or event.inaxes != self.ax or event.xdata is None:
            return
        self.sensors[self.dragging].position[0] = event.xdata
        self.sensors[self.dragging].position[1] = event.ydata
        self.redraw()

    def on_release(self, event) -> None:
        if self.dragging is None:
            return
        moved = False
        if self.press_xy is not None and event.xdata is not None:
            moved = np.hypot(event.xdata - self.press_xy[0], event.ydata - self.press_xy[1]) > 1e-6
        if not moved:
            self.select(self.dragging)
        self.dragging = None
        self.press_xy = None

    def select(self, idx: int) -> None:
        self.selected = idx
        polar, azimuth = _polar_azimuth_from_rotation(self.sensors[idx].orientation)
        self._suppress_selected_callback = True
        self.selected_polar_slider.set_val(np.degrees(polar))
        self.selected_azimuth_slider.set_val(np.degrees(azimuth))
        self._suppress_selected_callback = False
        self.redraw()

    def on_selected_angle_change(self, _val) -> None:
        if self._suppress_selected_callback or self.selected is None:
            return
        polar = np.radians(self.selected_polar_slider.val)
        azimuth = np.radians(self.selected_azimuth_slider.val)
        sn = self.sensors[self.selected]
        from kresling_sim.sensors import tilt_rotation, Sensor as _Sensor

        self.sensors[self.selected] = _Sensor(position=sn.position, orientation=tilt_rotation(polar, azimuth))
        self.redraw()

    def on_save(self, _event) -> None:
        path = self.filename_box.text.strip() or "sensor_layout.npz"
        save_sensors(self.sensors, path)
        self.status_text.set_text(f"saved {len(self.sensors)} sensors -> {path}")
        self.fig.canvas.draw_idle()


def _polar_azimuth_from_rotation(rotation) -> tuple[float, float]:
    """Inverse of sensors.tilt_rotation: recover (polar, azimuth) radians
    from a Rotation produced by it, for displaying a sensor's current tilt
    on the per-sensor sliders."""
    local_z = rotation.apply([0.0, 0.0, 1.0])
    polar = float(np.arccos(np.clip(local_z[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(local_z[1], local_z[0])) if polar > 1e-9 else 0.0
    return polar, azimuth


def main() -> None:
    gui = SensorGUI()
    plt.show()


if __name__ == "__main__":
    main()
