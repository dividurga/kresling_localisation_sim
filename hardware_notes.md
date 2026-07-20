# Hardware Notes — Kresling Robot + Helmholtz Coil

Extracted from `background-lit/` (sibling directory, outside this repo):
- `Nature 2025_Main text.pdf` + `Nature_2025_SI.pdf` — "Modular chiral origami
  metamaterials", Nature 640, 2025 (Zhao, Dang, Manos, Zang, Mandal, Chen, Paulino).
  Princeton.
- `IEEE.pdf` — "Wireless Actuation of Magnetic Robots with a Modular 60 mT 3-D
  Helmholtz Coil System", IEEE APEC 2025 (Manos, Rao, Zhao, Liu, Zhou, Nguyen, Chen,
  Paulino, Chen). Same Princeton group — this is the coil system used to actuate the
  Kresling robots.

## Localization-target magnet

The thing v1 is tracking (per TODO.md scope: single Kresling-cell magnet, pose
of that one magnet):

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
