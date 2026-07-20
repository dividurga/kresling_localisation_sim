# Kresling Localization Sim — v1 Roadmap

Confirmed scope lives in Claude memory (`project_v1_scope.md` / `hardware_facts_kresling.md`)
and in prior conversation. Summary of what's locked in:

- Target: single Kresling-cell magnet (one 2mm x 0.5mm N50 cylinder, out-of-plane
  magnetized) = pose (position + orientation) of that magnet. No per-cell tracking of
  multiple cells.
- Actuation field OFF during sensing (no Helmholtz background term in v1).
- Sensor array: one horizontal plane below the workspace, FIXED axis-aligned orientation
  (all sensors face the workspace) — position-only grid sweep. Tilt optimization is v1.1.
- Footprint bound: plate footprint TBD (open item, see below) + min sensor pitch (~8mm
  placeholder, MLX90393-class QFN — confirm/adjust).
- **Workspace z-range: ~3mm to ~28.3mm.** Locomotion mode has the robot rolling on its
  own edges close to the actuation surface (same behavior as the IEEE-paper cube robot
  rolling on its faces) — farthest it lifts off is bounded by its own deployed cell
  height h1=28.3mm. Sensor plane sits ~3mm below the surface the robot actually moves on
  (mechanical min gap).
  On-axis field over this z-range spans ~11.4 mT (at 3mm) down to ~13 uT (at 30mm) for a
  single magnet directly above a sensor -- good SNR throughout, no saturation risk at the
  near end on-axis (off-axis/closer-than-3mm cases not yet checked).
- Because locomotion = rolling, the magnet's orientation sweeps through a wide range of
  angles relative to the sensor plane as the robot rolls (not just translating with fixed
  orientation) -- orientation variation is in-scope for the forward model / CRB sweep,
  not an edge case.
- magnet: NdFeB N50, dia 2mm x 0.5mm, out-of-plane magnetization. Br ~1.40-1.45T
  (not stated in paper, standard N50 datasheet value) -> compute moment m = Br*V/mu0
  in code, don't hand-calc.
- Sensor: representative 3-axis Hall IC, MLX90393-class (~50mT range, placeholder noise
  floor) - swap for real datasheet numbers later.
- Library: magpylib for forward field model.
- Optimality metric: Cramer-Rao Bound (CRB / Fisher info) for fast search, Monte Carlo
  (noisy nonlinear least-squares inverse solve) to validate top candidates.

## Files scaffolded (all currently empty)

- `src/kresling_sim/dipole.py` — magpylib dipole source wrapper, parametrized by pose
  (x,y,z, orientation) given known moment magnitude |m|.
- `src/kresling_sim/sensors.py` — planar sensor grid: given plate footprint, pitch, and
  sensor count/layout, produce sensor positions (fixed axis-aligned orientation in v1).
- `src/kresling_sim/forward.py` — glue: dipole + sensor grid -> predicted B field
  (3 components) at each sensor.
- `src/kresling_sim/fisher.py` — Jacobian of stacked sensor readings w.r.t. pose unknowns,
  Fisher info = J^T Sigma^-1 J, CRB = diag(inverse). Sweep over grid configs.
- `src/kresling_sim/inverse.py` — nonlinear least-squares pose recovery from noisy
  readings (scipy.optimize.least_squares), using the same forward model.
- `src/kresling_sim/montecarlo.py` — draw noise + pose samples, run inverse solver many
  times, compare empirical RMS error to CRB bound for top-ranked configs.
- `scripts/sweep.py` — top-level script: sweep sensor count/spacing, run CRB + MC,
  plot error vs sensor count to pick "optimum" placement.

## Suggested build order (do these one at a time, test as you go)

1. [ ] `dipole.py` — wrap a magpylib dipole/magnet, taking pose params in, returning a
       magpylib source object (or a function pose -> source).
2. [ ] `sensors.py` — planar grid generator: inputs = footprint size, pitch/count,
       standoff height; output = array of 3D sensor positions (all same fixed
       orientation, e.g. sensitive z-axis normal to plate).
3. [ ] `forward.py` — for a given dipole pose + sensor grid, compute predicted
       [Bx,By,Bz] per sensor via magpylib's `getB`.
4. [ ] Sanity check forward.py manually: does field falloff / magnitude look physically
       reasonable across the ~3-28.3mm z-range given the tiny 2mm magnet moment?
       This is a good moment to double check the Br/moment assumption before building
       further on top of it.
5. [ ] `fisher.py` — numeric (finite-difference) Jacobian of forward model w.r.t. pose,
       assemble Fisher info matrix using assumed sensor noise covariance, invert for CRB.
6. [ ] `inverse.py` — scipy least_squares wrapper: noisy simulated reading -> recovered
       pose estimate.
7. [ ] `montecarlo.py` — loop inverse.py over many noise draws + pose samples for a
       given sensor config, collect RMS error stats.
8. [ ] `scripts/sweep.py` — sweep sensor count/spacing (respecting footprint + pitch
       constraints), rank by CRB, validate top candidates with montecarlo.py, plot.

## Open items to revisit before/while building

- [ ] Confirm min sensor pitch (currently placeholder ~8mm) and exact Hall IC part/specs.
- [ ] Confirm exact plate footprint (x,y) dimensions -- this is now an open question about
      how far/wide the robot roams horizontally during locomotion (separate from the
      z-range, which is now settled at ~3-28.3mm).
- [ ] Decide what "representative robot poses spanning the workspace" means concretely
      for the CRB sweep -- now needs z sampled over ~[3mm, 28.3mm] and orientation sampled
      over the range actually seen while rolling (not just uniform random over all
      orientations) -- may want to look at the IEEE cube-rolling kinematics or Nature Fig 6a
      locomotion description again to bound the realistic orientation sweep.
- [ ] Check saturation risk for off-axis / closer-than-3mm cases (on-axis at 3mm was ~11.4mT,
      within the placeholder +/-50mT sensor range, but not yet checked off-axis or for
      distances below the nominal 3mm gap).
