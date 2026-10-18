tldr: what we actually found

I think I may have had a huge oversight.

I was looking at videos of the system, and realised its not a single magnet. 

Can I be shown the actuation??
Setup:
  - sensor array is fixed on the floor, kresling cell moves above it
  - workspace is 100x100mm footprint, 6.35mm to 31.65mm in z (PCB standoff to fully deployed)
  - sensors are MLX90393 hall sensors, noise sigma_xy=0.0008mT, sigma_z=0.0014mT
  - magnet is a tiny 2mm dia x 0.5mm N50 disk on the kresling cell
  - we swept sensor count (5x5 to 10x10 grids) and pitch (12-24mm spacing)

How we evaluated:
  - CRB (Cramer-Rao Bound): analytical lower bound on localization error. fast, no solver needed.
    tells you the best any unbiased estimator can possibly do given the sensor noise.
  - Monte Carlo: actually ran the nonlinear solver 30 times per position with warm-start
    (initial guess perturbed by ~3mm from true position, simulating tracking mode).
    this is what youd actually get in the real system.

Why warm-start:
  - cold-start (solver starts from centroid of sensor array) completely fails at workspace edges.
    the dipole field has local minima, solver finds wrong answer when it starts 76mm away.
  - warm-start simulates the real use case: youre tracking a moving cell, last known position
    is close to truth. this is realistic and actually works.

What we found:

  CRB says: more sensors + smaller pitch = better. 10x10 @12mm wins with mean=0.385mm.

  MC says: sweet spot is 7x7-8x8 @ 12-14mm, mean ~0.6-0.74mm.

  The gap (MC/CRB ratio):
    - small grids (5x5, 6x6): ratio ~1.05-1.26. CRB is a good predictor here.
    - big grids (9x9, 10x10): ratio up to 1.6. CRB is too optimistic.
      more sensors = harder loss landscape = solver occasionally gets stuck. This is something that we need to research in
      what sort of reconstruction algo can we use to not get stuck

  Max error spikes:
    - some configs have MC max >> CRB max. eg 9x9 @16mm: CRB max=0.51mm, MC max=6.98mm.
    - these are solver convergence failures at workspace corners, not physics.
    - 7x7 @14mm stays clean: CRB max=1.01mm, MC max=1.48mm. no scary spikes.

Recommendation:
  7x7 @ 14mm.
    - 49 sensors, 14mm pitch, 84mm span (covers the 100mm workspace with a bit of overhang)
    - mean MC RMSE ~0.64mm, max ~1.48mm
    - CRB/MC ratio 1.24 -- bound is fairly tight, solver is reliable
    - going bigger (9x9, 10x10) buys you maybe 0.05mm on mean but adds solver instability
    - going smaller (5x5, 6x6) gets noticeably worse at workspace corners

  If you really want the absolute best and can afford 64+ sensors: 8x8 @12mm.
    - mean MC RMSE ~0.59mm, 64 sensors, 84mm span
    - still has some max spikes (1.66mm) but manageable

GUI:
  We can configure the following:
    - 

Second Setup, with two magnets:
    - From the CRB sweep and monte-carlo with chosen configs, i think the way to go is 6x6 @ 18 mm pitch
    - assuming this, if we do an analysis on min conversion time to calculate frequency of localisation, here are our findings:


    setup: hex prism, two parallel N42 4mm x 0.5mm discs 12mm apart, axis along +y, resting flat so z is 11.3-13.4mm (12.35mm +/- 10deg of rotation)

    sensor array (6x6 @ 18mm, 36 sensors, 90mm span):
      - CRB mean 0.036mm, worst position 0.080mm (9x9 xy x 4z grid, noise at 14ms conversion)
      - MC (multi solve, n_starts=5, warm start 3mm, 5x5 xy x 4z grid, 20 trials): 0.0441mm vs CRB 0.0445mm on that same grid, MC/CRB ~0.99. shortlist run converged 100%
      - 25 sensors (5x5 @ 20mm) is ok on mean but worst position is 0.15mm. past 36 you get maybe 10% per extra step
      - bigger pitches (6x6 @22, 7x7 @18, 8x8 @16) hang over the 100mm workspace
      - still need to check the bus can do 36 sensors (spi cs / i2c mux)
      - data in scripts/sensor_array/kresling_sweep_multi_solve/ (pareto_sensors_vs_error.png, per config folders)

    the sensing window:
      - sequence is coils off -> wait for the coil field to die -> one mlx conversion -> coils back on
      - wait until residual coil field < sigma/3 on every axis. coils are modelled first order (actuation.py), tau = t_fall/ln9
      - fall time 5ms is the worst case (controller design target, sci adv 2026 SI eq S19). the 1.5ms in the IEEE paper is a measured step response not the design bandwidth
      - at 20A the coil field at the sensors is 13-14mT, so the wait is ~4.9 x fall time. 5ms fall -> 21-28ms wait whichever conversion time you pick
      - fall time is the lever not conversion time. 1.5ms fall would give a ~7ms wait (scripts/fall_time_analysis.py). nobody has measured the real fall yet

    what the noise implies (5ms fall, 20A, 6x6 @ 18mm, 10ms actuation window):

| conv (ms) | sigma xy / z (uT) | wait (ms) | off time (ms) | rate (Hz) | CRB x / y / z (mm) | CRB mean | CRB worst |
|---|---|---|---|---|---|---|---|
| 1.3 | 5.10 / 7.60 | 20.6 | 21.9 | 31.3 | 0.194 / 0.119 / 0.107 | 0.253 | 0.496 |
| 2 | 3.30 / 4.90 | 21.6 | 23.6 | 29.8 | 0.125 / 0.077 / 0.069 | 0.164 | 0.321 |
| 3 | 2.45 / 4.00 | 22.2 | 25.2 | 28.4 | 0.095 / 0.059 / 0.052 | 0.124 | 0.242 |
| 5 | 1.60 / 2.70 | 23.2 | 28.2 | 26.2 | 0.062 / 0.039 / 0.034 | 0.082 | 0.159 |
| 8 | 1.15 / 2.00 | 24.0 | 32.0 | 23.8 | 0.045 / 0.029 / 0.025 | 0.059 | 0.115 |
| 14 | 0.80 / 1.45 | 24.8 | 38.8 | 20.5 | 0.031 / 0.020 / 0.017 | 0.041 | 0.081 |
| 25 | 0.60 / 1.05 | 25.4 | 50.4 | 16.6 | 0.023 / 0.015 / 0.013 | 0.031 | 0.060 |
| 50 | 0.45 / 0.80 | 26.0 | 76.0 | 11.6 | 0.018 / 0.011 / 0.010 | 0.023 | 0.045 |
| 100 | 0.30 / 0.55 | 27.0 | 127.0 | 7.3 | 0.012 / 0.008 / 0.006 | 0.016 | 0.030 |
| 175 | 0.20 / 0.40 | 28.0 | 203.0 | 4.7 | 0.008 / 0.005 / 0.004 | 0.011 | 0.021 |

    how to read it:
      - position error scales ~linearly with sigma. every 2x on conversion time is ~1.4x less error but costs the full extra ms in the window
      - x is the worst axis, y and z are about 1.5-2x better (didnt dig into why)
      - with 5ms fall the wait is ~22ms of the window either way, so the 2-5ms rows are cheap, 14ms+ starts to cost real rate
      - 2ms conversion: ~30Hz, 0.16mm mean. 5ms: ~26Hz, 0.08mm mean. 14ms: ~20Hz, 0.04mm mean
      - the roughly 0.65mm y bias from an 8deg tilt mismatch (divijas_notes.md) is bigger than every row here (worst mean is 0.25mm). nailing the resting tilt matters more than conversion time
      - plots: scripts/noise_implications.png, scripts/conversion_time_experiment.png, scripts/fall_time_analysis.png

    caveats:
      - sigma is read off the datasheet chart (rev 012 figs 2 and 3, midpoint of the range, +-0.1uT), not measured
      - noise assumed white, zero mean, gaussian, independent across axes and sensors. no offset/gain error, no sensor misplacement, no magnet moment error
      - CRB is a floor, MC tracked it to ~1-2% on the configs we ran but only warm start (3mm), never cold start
      - fall assumed first order and the same as the rise. real driver could be faster (active sink) or slower (kick back, eddy tail)
      - readout time over spi/i2c is not in the off time
      - i also haveny really looked into delay between sensors
      - or just general delays and communication-level constructs that will inevitably come up when we have like 36 sensors being controlled by the same mcu (cooked? @konstantinos what will this work entail?)
      - 20A only. 100A adds a few ms and goes past the sensor saturation onset (~50mT)

    scripts: conversion_time_experiment.py, fall_time_analysis.py, noise_implications.py, pareto_array.py (all in scripts/)
