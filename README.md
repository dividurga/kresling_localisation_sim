# kresling-localisation-sim

Simulation for Hall-sensor-based magnetic localization of a Kresling-cell
robot's on-board magnet, driven by a 3-D Helmholtz coil actuation system.
Covers forward field modeling (`kresling_sim.coils`, `kresling_sim.kresling`),
sensor noise (`kresling_sim.noise`, `kresling_sim.sensors`), CRB-based sensor
layout ranking (`kresling_sim.fisher`), and pose solving (`kresling_sim.inverse`,
`kresling_sim.montecarlo`). See `hardware_notes.md` for the hardware/paper
context this is built against.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root
uv sync
```

`uv sync` reads `pyproject.toml`/`uv.lock` and creates a `.venv` with all
dependencies (`magpylib`, `matplotlib`, `numpy`, `scipy`) pinned to the
locked versions. No separate `requirements.txt` — `uv.lock` is the source
of truth; run `uv lock --upgrade` if you want to bump pinned versions.

## Running things

Prefix any script with `uv run` so it executes inside the project's venv
without needing to activate it manually:

```bash
# sanity checks
uv run python tests/sanity_check_coils_actuation.py

# plots / sweeps
uv run python scripts/field_falloff_check.py
uv run python scripts/actuation_cycle_check.py
uv run python scripts/sweep.py
```

Each script's docstring documents its own flags and defaults — pass `--help`
for the full list, e.g. `uv run python scripts/field_falloff_check.py --help`.

To add a new dependency: `uv add <package>` (updates `pyproject.toml` and
`uv.lock` together — don't hand-edit either).
