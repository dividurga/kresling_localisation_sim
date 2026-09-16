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