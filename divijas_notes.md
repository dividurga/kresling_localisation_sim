stuff i still need to go check on, written up so i don't lose track. not code, not "final" anything, just the open threads from working through the two-magnet / kresling_2026_supp stuff.

## the resting geometry thing is still the big one

so the truss cell isn't a straight prism, it's twisted -- mountain rods are 10.5mm and valley rods are 12.8mm (table S1), which only makes sense if the top and bottom hex plates are rotated relative to each other. means when the thing is lying on its side for sliding there's no flat face to rest flush on, contact is probably a rod or a plate vertex, and which one (plus how high the axis sits above it) depends on which way it happens to have settled. paper never has to deal with this because their sliding force equations are 1D in y and never touch z at all -- they're not doing hall sensor stuff, they track with cameras.

the standoff number in kresling_2026_supp_config.py (Z_MIN/Z_MAX, 7.35-18.35mm) is a deliberately wide placeholder because of this, not a real derivation. widened it on purpose instead of pretending precision we don't have. would love a real measurement or an actual statics derivation of the resting pose at some point.

related, and i almost missed this: if it rests tilted even a little (not perfectly level), the two magnets end up at slightly different z too, not just the whole assembly at some standoff -- z_magnet ~ z_center +/- (h/2)*sin(tilt). same root cause (twisted truss, unknown resting pose), just a second symptom of it. SLIDING_ORIENTATION in the config is currently pure horizontal (zero tilt) as a simplification. don't have a real tilt angle to put in there. architecture already supports it whenever we do (just change the orientation constant, no code changes needed) -- separation_mm + orientation already compose correctly for any tilt, verified this when building it.

correction to the above, tested it properly instead of assuming: a wrong tilt assumption does NOT stay contained to the z-estimate. ran a real experiment -- true magnet tilted 8deg off horizontal, solver assumes zero tilt (like the config currently does), solve for position only. error came out (x,y,z) = (0.052, 0.645, -0.303)mm. x basically untouched, but y moved MORE than z did. reran with the tilt confined to a different plane (x-z instead of y-z) and the corrupted coordinates flipped to match -- x+z corrupted, y clean that time. so the rule is: whichever plane the orientation-mismatch rotation lives in, BOTH coordinates in that plane get biased, the third (orthogonal) one stays close to clean. for our sliding setup specifically, the tilt can only ever live in the y-z plane (tilt_rotation's rotation axis is x at our azimuth=pi/2 convention), so y is NOT protected the way i assumed -- a wrong SLIDING_ORIENTATION guess leaks straight into the y number, which is the one number an eventual sliding controller would actually be reading. checked this isn't a coincidence by computing it two ways: full nonlinear least-squares solve, and a linearized version using fisher.position_jacobian directly (find the position shift whose field pattern best explains the leftover field difference between true and assumed orientation) -- both gave the same answer, (0.068, 0.727, -0.293) predicted vs (0.052, 0.645, -0.303) actual, close enough that the mismatch is just higher-order effects from the tilt not being infinitesimally small. mechanism, in short: a small wrong-tilt is equivalent (to first order) to the magnet secretly having a bit of extra moment pointing in z that our model doesn't know about, and the solver -- not allowed to touch orientation -- fakes that missing signal by nudging position instead, borrowing from whichever position directions (y, z) can actually mimic that kind of field change. x can't mimic it, so x stays clean; y and z can, so they don't. bumps up the priority on nailing down the resting tilt, not just the standoff distance.

decided: made the tilt actually configurable instead of hardcoded at zero. new constant in kresling_2026_supp_config.py, RESTING_TILT_DEG (currently 0.0, placeholder), SLIDING_ORIENTATION now derives from it (tilt_rotation(pi/2 - radians(RESTING_TILT_DEG), pi/2)) instead of hardcoding pi/2. going to go check the actual robot/experimental setup and figure out what this angle really is, then just update the one number. everything downstream (both magnets' z, the whole forward model) already composes correctly off this one constant, verified -- no other code needs to change when the real value comes in.

still only localising the COM (x,y,z), not the tilt itself -- tilt stays a known/configured constant fed into the forward model, same as separation_mm always has been, not something the solver estimates. that was the right call before and still is: adding tilt as a 4th unknown would be real new scope (basically the deferred orientation-solve extension), and we don't need it if we can just go measure the actual resting angle instead.

## hall sensor noise operating point needs redoing

found the coil rise time was wrong (1.5ms, should've been 5ms per the sci adv SI's own eq S19) and fixed that in actuation.py/config.py. but the ~10-14ms conversion time choice for the MLX90393 was picked specifically to clear that wrong 1.5ms transient. with the real 5ms transient this tradeoff needs re-deriving, probably pushes to a longer conversion time (~25ms row on the noise table) which means a lower noise floor but a longer sensing window. haven't touched this yet, still using the old sigma_xy/sigma_z values in DEFAULT_NOISE.

## magnet Br is paper-sourced now but still not datasheet-verified

MAGNET_BR_MT in kresling_2026_supp_config.py = 1319.5mT, computed from the paper's own M0=1.05e6 A/m (table S9), so this is better than a guess. but still worth checking against an actual N42 datasheet before trusting it for anything that matters, same flag the old N50 value already carried in config.py. nobody's done that yet.

## does the geometric midpoint between magnets actually equal the paper's Y_O (COM)?

argued yes because two identical magnets means their COM is the midpoint, and the paper's own Y_O +/- h/2 formula implies the same midpoint. but that's only the FULL robot's COM if the rest of the truss frame (rods + plates) is ALSO mass-symmetric about that same plane, which the paper doesn't explicitly say, just strongly implies via the symmetric-looking construction. probably fine but worth being honest it's an assumption, not a confirmed fact.

## tangential orientation sign convention (circular array feature, unrelated to two-magnet stuff)

from way earlier -- circular_array.py's tangential mode uses theta+90deg (CCW) as an arbitrary placeholder, never checked against an actual PCB layout convention. still open, low priority, easy one-line flip whenever it matters.

## locomotion modes we haven't touched

only sliding is modeled right now (deliberately, per the "orientation stays fixed" simplification that let us skip an orientation-solve extension). rolling/steering/walking all need z (and full orientation) to actually vary over time, which is real new scope -- the deferred v1.1 tilt-optimization work, not a two-magnet extension. not started.

## sweep resolution -- run a second, purpose-tuned sweep after this one

first characterise_sensor_arrays.py run for kresling_2026_supp is using the exact same test grid as kresling/cube_robot (N_XY=5, N_Z=4, full 100x100mm xy square, 25mm spacing) since that script isn't parametrized per target. keeping this run as-is on purpose -- it's a same-protocol baseline directly comparable to the other two targets, useful for the paper even though it's not tuned for this target specifically.

for next time: this target's CRB numbers are ~10x tighter than kresling/cube_robot (0.03-0.3mm vs the 0.1-7mm range), so a 25mm-spaced xy test grid probably misses real spatial structure, especially at the tighter 12mm-pitch sensor configs. also sliding is directional (travels along y, fixed orientation) not omnidirectional like rolling/cube_robot, so testing the full square footprint might not even be the right region -- a narrower strip along the direction of travel would be more representative. plan: denser xy grid + travel-direction-aware footprint, as a second sweep once this baseline one finishes.

## experimental finding: ghost-solution failure mode in the two-magnet (in-plane) forward model

**summary**: `characterise_sensor_arrays.py` for kresling_2026_supp turned up Monte Carlo RMSE up to 200x worse than the analytical CRB on some configs (sparser/wider-pitch ones, 7x7 @18-22mm first noticed). not noise, not a crash -- individual trials genuinely converging to the wrong answer with high confidence. this is real, reproduced multiple ways below, and specific to this target's in-plane (horizontal) magnetization -- the old vertical-magnet targets don't have it.

**where it happens**: exactly when the true source's coordinate perpendicular to the magnetization direction lines up with a sensor row/column. for our default Y-magnetized setup, that means the source's X coincides with a sensor column's X. tested x=0,20,-20,-40,-60,60 (all on-lattice for a 7x7@20mm grid) -- all bad to varying degrees (~5-30% of trials). x=10 (exactly between columns) -- clean, 0 bad trials in every batch tested. confirmed this is genuinely about lattice alignment, not "the origin is special": local Fisher-information conditioning at x=0 and x=20 is identical (same eigenvalues), so nothing about x=0 itself is locally different from x=20 -- both just happen to sit on a column.

**confirmed the mechanism is tied to magnetization direction, not anything Y-specific**: rotated the magnetization from along-Y to along-X and reran. the failure axis flipped exactly as predicted -- x=0 became clean, y=0/y=20 (now row-aligned) became bad instead, y=10 stayed clean. rules out anything incidental to "Y" -- it's specifically "source position perpendicular-to-magnetization coincides with the sensor lattice."

**confirmed the old vertical single-magnet target doesn't have this**: same 7x7@20mm grid, same lattice-aligned x positions -- zero bad trials anywhere tested. so this is genuinely new, introduced by choosing in-plane magnetization for this target, not something that was always lurking unnoticed.

**why vertical is immune and horizontal isn't (verified, not just argued)**: every dipole-like source has a line of rotational symmetry through it along its own moment direction. for vertical magnetization that axis points straight up/down -- perpendicular to the flat sensor array, never parallel to any row or column no matter where the source sits. for horizontal magnetization, that axis runs horizontal, in the same plane the array's rows/columns run in -- when the source's position lines up with one, that whole column ends up coplanar with the source's own symmetry axis and loses some ability to distinguish positions around it.

**one exact, provable piece of this**: for a sensor sharing the source's x-coordinate, with moment along y, `Bx = 0` exactly (not approximately -- verified to floating-point precision, ~1e-17, on the real Cylinder-based model). derivation: with Δ=(0,y-y0,z-z0) and m=(0,m,0), both m̂ and r̂ have zero x-component, so `Bx = 3(m̂·r̂)r̂ₓ - m̂ₓ = 0`. turns out this identity holds for vertical magnetization too (checked directly), so it's necessary but not sufficient -- it's not the full distinguishing fact by itself.

**honest limit of the symbolic derivation**: pushed further trying to find a single closed-form reason the aggregate (all 7 columns, noise-weighted) landscape goes flat for horizontal and not vertical. leading far-field decay rate along the column turns out to be the *same* order (1/u³) for both orientations' dominant field component -- ruled that out too. what's left is a repeatedly-confirmed *empirical* fact (the phi-landscape numbers below, the FIM eigenvalue collapse, the swap test), not a tidy one-line proof. being honest about that rather than overclaiming rigor i don't have.

**concrete numbers -- the flat landscape**, noiseless phi(x=0,y=dy,z=z_true) sweep:
```
horizontal:  dy=5->2.5e6  dy=10->1.6e6  dy=20->2.3e5  dy=40->4.9e5  dy=60->5.0e5  dy=80->3.9e5  dy=100->3.1e5
vertical:    dy=5->1.7e3  dy=10->3.8e3  dy=20->6.9e3  dy=40->8.1e3  dy=60->7.5e3  dy=80->4.2e3  dy=100->3.9e3
```
horizontal: steep wall right near truth, then a shelf that barely changes across dy=20-100 (factor of ~2). vertical: smooth rise-then-fall, no comparable flat shelf.

**concrete numbers -- the ghost is a genuinely bad fit, not a competing good one**: phi(true_pos)=0, phi(one observed ghost at (-0.13, 83.6, -27.4))=311,227. FIM eigenvalues collapse near the ghost too (7-18 vs 3658-276038 near truth, so 500-25,000x weaker sensitivity in every direction) -- the solver isn't finding a good alternative, it's getting stuck on a plateau so flat that its own step-size/cost-change stopping criteria trigger long before it's found anything good. `SolverResult.SUCCESS` just checks scipy's internal convergence flags, never checks the residual is actually small.

**concrete numbers -- the basin is razor thin**: same noiseless measurement, started the solver from increasing distances along y from the true position:
```
0.5-5.0mm away  -> exact truth every time
6.0mm away      -> 7255mm off
7.0mm away      -> 10185mm off (a DIFFERENT wrong point)
8.0-15.0mm away -> the same ~84mm ghost, consistently
20.0mm away     -> yet another different wrong point
```
basin edge sits between 5 and 6mm. warm_start_sigma=3mm means a ~2-sigma draw (not rare, ~5% per trial) lands past that edge. that's the concrete reason warm-starting doesn't reliably save us here.

**fixed**: `LeastSquaresSolver` (`inverse.py`) now takes `n_starts` (default 1, backward compatible -- verified byte-identical to old behavior for existing targets), `start_spread_mm` (default 20.0), `max_residual_ratio` (default 5.0). runs `n_starts` independent solves (extra ones jittered around the initial guess), keeps the lowest-cost converged attempt, AND rejects it outright (returns FAILURE) if its cost still exceeds `max_residual_ratio` times the expected noise-only cost (0.5*3*n_sensors, chi-squared expectation for a correct fit). `PositionSolver.solve()` and `montecarlo.py`'s call site now thread an `rng` through for reproducible jitter.

turns out the residual-sanity-check is doing most of the actual work, not the extra starts. re-tested the exact worst position (0,0,7.35mm), 7x7@20mm: at n_starts=1 (jitter off, only the safeguard active), converged dropped from a false 15/15 "success" to an honest 23/30 -- the bad trials now correctly report FAILURE instead of silently returning garbage, so rmse over the converged ones came out 0.0173mm, right on top of crb=0.0170mm. bumping to n_starts=5 recovered more of the lost trials: 27/30 converged, rmse=0.0138mm. so the safeguard alone stops garbage from poisoning the aggregate; the extra starts on top of it claw back some of the otherwise-lost convergence rate.

confirmed at full-sweep scale, not just the one position: reran all 5 previously-worst configs (7x7 @14/16/18/20/22mm, same 100-position test grid as the original characterise_sensor_arrays.py sweep, n_starts=5) --

```
config       before(MC/CRB)   after(MC/CRB)
7x7 @14mm    224%             100%
7x7 @16mm    420%             100%
7x7 @18mm    408%             101%
7x7 @20mm    1008%            99%
7x7 @22mm    1060%            101%
```

every one of them, essentially perfect agreement now. fix confirmed, not just at the hand-picked worst position but across the actual sweep configs that originally exposed the problem. still need to decide whether to rerun the FULL 42-config characterise_sensor_arrays.py + mc_spatial_heatmaps.py + sweep.py with the fixed solver (their existing completed runs all predate this fix and carry the contaminated data on the affected configs), and whether to bump characterise_sensor_arrays.py's default solver to n_starts>1 so future runs get this for free instead of needing to remember to pass it explicitly.

## the render script

scripts/render_setup_2026_supp.py -- at true scale the magnets are basically invisible dots against the coil system, which is correct/expected, not a bug, just worth remembering when looking at it so it doesn't look "wrong."
