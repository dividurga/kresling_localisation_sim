## AI-assisted parsing of facts about HW so we have a point of reference without having to read the papers again
# Hardware Notes — Kresling Robot + Helmholtz Coils


Extracted from `background-lit/` (sibling directory, outside this repo):
- `Nature 2025_Main text.pdf` + `Nature_2025_SI.pdf` — "Modular chiral origami
  metamaterials", Nature 640, 2025 (Zhao, Dang, Manos, Zang, Mandal, Chen, Paulino).
  Princeton.
- `IEEE.pdf` — "Wireless Actuation of Magnetic Robots with a Modular 60 mT 3-D
  Helmholtz Coil System", IEEE APEC 2025 (Manos, Rao, Zhao, Liu, Zhou, Nguyen, Chen,
  Paulino, Chen). Same Princeton group — this is the coil system used to actuate the
  Kresling robots.

## Localization-target magnet

The thing v1 is tracking: single Kresling-cell magnet, pose of that one
magnet:

- Small cylindrical NdFeB **N50** magnet, **⌀2 mm × 0.5 mm**, out-of-plane
  magnetization, glued (Loctite Super Glue Ultra Gel) at the top rim/vertices of
  each Kresling cell (Nature SI, Fig. S15a).
- No Br or magnetic-moment value is stated in the paper — N50 datasheet Br is
  typically ≈1.40–1.45 T. Compute moment m = Br·V/μ₀ in code (don't hand-calc and
  hardcode it — verify the number once it's computed).

A second, unrelated magnet type also appears in the papers but is **not** the
localization target per the confirmed v1 scope: 16mm×2mm square soft-magnetic
composite plates (Nd-Pr-Fe-B powder in Ecoflex silicone), in-plane magnetized,
used on top/bottom tessellation plates for "transformer mode" locomotion — much
weaker field, different geometry.

## Robot/cell geometry

6-gon Kresling cell used in the paper:
- polygon radius r = 16.8 mm
- deployed height h1 = 28.3 mm, folded height h0 = 12.6 mm
- max twist angle Δφ = 83° (kinematic limit <90° from the rotating-square
  tessellation)
- Multi-cell assemblies scale via α = cos(φ/2) (in-plane) and β = height ratio,
  but no absolute full-assembly L/H is given numerically in either paper — only
  ratios/formulas. Photos show assemblies from 3×3 up to 7×7×7 cells.

## Locomotion / workspace scale

In "locomotion mode" the robot moves as a rigid body by rolling on its own edges
close to the actuation surface (Nature main text, Fig 6a: "locomotion mode
(robotic motion, rigid body)") -- the same rolling-on-faces behavior as the
cubic robot in IEEE.pdf. So the robot's z-height above the surface is bounded by
its own physical size:

- z-range ~3mm (mechanical min gap between sensor plane and the surface the robot
  moves on) to ~28.3mm (h1, the deployed single-cell height -- the farthest a
  point on the cell lifts off while rolling).
- On-axis field from the single 2mm x 0.5mm N50 magnet over this z-range: ~11.4mT
  at 3mm down to ~13uT at 30mm -- good SNR throughout for a realistic Hall IC, no
  saturation risk at the near end on-axis (off-axis / sub-3mm cases unchecked).
- Because the robot rolls (not just translates), the magnet's orientation sweeps
  through a wide range of angles relative to the sensor plane during motion --
  this is in-scope for the forward model and CRB sweep, not an edge case to
  ignore.
- The (x,y) footprint the robot roams horizontally is still an open question,
  separate from the z-range -- not yet pinned down.

## Hall sensor noise floor (MLX90393)

Source: Melexis MLX90393 datasheet, Rev 012 - Nov 18, 2025
(https://www.melexis.com/-/media/files/documents/datasheets/mlx90393-datasheet-melexis.pdf),
section 13.1 "Noise vs Conversion Time", Figures 2 (XY-axis) & 3 (Z-axis),
GAIN_SEL=0x7. Noise is not a single spec -- it's a log-log tradeoff curve
against conversion time, at 4 discrete OSR settings; values below are read
off the chart's marker positions (not a printed table), so treat as
+-1 mGauss precision. Z-axis noise is consistently higher than XY at every
conversion time (not specific to one operating point).

| Conversion time | XY noise | XY noise | Z noise | Z noise |
|---|---|---|---|---|
|  | (mGauss) | (uT) | (mGauss) | (uT) |
| ~1.3 ms | 51 | 5.1 | 76 | 7.6 |
| ~2 ms | 31-35 | 3.1-3.5 | 46-52 | 4.6-5.2 |
| ~3 ms | 23-26 | 2.3-2.6 | 35-45 | 3.5-4.5 |
| ~5 ms | 14-18 | 1.4-1.8 | 24-30 | 2.4-3.0 |
| ~8 ms | 11-12 | 1.1-1.2 | 17-23 | 1.7-2.3 |
| ~14 ms | 8 | 0.8 | 12-17 | 1.2-1.7 |
| ~25 ms | 6 | 0.6 | 9-12 | 0.9-1.2 |
| ~50 ms | 4.5 | 0.45 | 7-9 | 0.7-0.9 |
| ~100 ms | 3 | 0.3 | 5-6 | 0.5-0.6 |
| ~150-200 ms | 2 | 0.2 | 4 | 0.4 |

**Chosen operating point (confirmed with user 2026-08-02)**: ~10-14ms
conversion time -- middle ground between sensing-window length (actuation
channels settle with ~1.5ms rise time per actuation.py, so the window needs
to clear that transient) and noise floor. sigma_xy ~= 0.0008 mT (0.8uT),
sigma_z ~= 0.0014 mT (1.4uT) -- wired into `montecarlo.DEFAULT_NOISE` as
`GaussianNoise([0.0008, 0.0008, 0.0014])`.

## Helmholtz coil (actuation, not sensing)

Per confirmed v1 scope, this field is assumed OFF during sensing — not modeled
in the v1 forward model. Kept here for context / v2 background-field work:

- 3 orthogonal coil pairs (x/y/z), each pair = inner+outer winding in series,
  10 AWG square copper wire.
- Coil radii ~10.0–16.5 cm, turns N=40–64 per winding, max field ~60 mT at
  origin, ~15 kW pulsed / up to 25kW modular converter, PI current control,
  bandwidth dc–25 Hz.
- Exact per-winding table (N, R, offset d, resistance, inductance) is in
  IEEE.pdf Table I / Nature SI Table S1 if needed later for a background-field
  model (v2+).
- No existing magnetic localization baseline in either paper — pose is tracked
  via motion-capture cameras + fiducial markers (OpenCV), not Hall sensors.
  Confirms the Hall-array approach is novel relative to this hardware group's
  own work.
