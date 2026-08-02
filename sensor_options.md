Okay so I went back and forth alot with my AI and read a bunch of datasheets, and the TLDR was to stick with the sensor chosen so far
bcs hopefully the multi sensor readings gives us high enough amount of information

Made the LLM make a summary that I'm adding down below for future reference

# Sensor Options — Addressing the Tight 28.3mm SNR Margin

Context: our on-axis field range at the sensor plane is **~11.4mT (3mm, closest)
down to ~15.7uT (28.3mm, farthest lift-off while rolling)** -- see
`scripts/field_falloff_check.py`. That's a **~726:1 dynamic range** (~9.5 bits)
the sensor has to cover: strong enough not to saturate up close, sensitive
enough to resolve the far end above its own noise floor. Using the MLX90393
(placeholder part) at a ~10ms conversion time, the far-end single-sensor SNR
works out to only ~15-16:1 -- workable, but tight. This doc looks at (1) other
sensor parts, and (2) what a stronger magnet would actually cost us.

## The core tension (read this before the table)

Every real magnetometer trades range for noise floor. In this search, that
split was stark and consistent:

- **Wide-range Hall ICs** (MLX90393, MLX90394, TLV493D): handle our 11.4mT
  near-field fine, but noise floors are uT-scale -- fine for the near end,
  tight at the far end.
- **Low-noise TMR/AMR/fluxgate parts** (AK09940A, BMM350, MMC5983MA, DRV425):
  noise floors are 10s of nT -- 10-100x better than the Hall ICs -- but their
  full-scale range tops out around **+/-0.8 to +/-2mT**. Our near-field alone
  (11.4mT at 3mm) would **saturate these instantly**, long before the robot
  even lifts off the plane. None of the low-noise parts found in this search
  have both wide range and low noise simultaneously -- that combination
  doesn't really exist off-the-shelf in a single fixed gain setting, which is
  exactly why parts like the MLX93 family expose a *programmable* gain/range
  instead of a fixed one.

**Before reaching for hardware:** the whole point of the CRB/Monte-Carlo
pipeline in `TODO.md` is that you don't rely on any *one* sensor's SNR --
you fuse many sensors' readings. Even if the single farthest-away sensor sees
only ~16:1 SNR, a dense enough array gives you many partially-independent
looks at the same event, and the Fisher-information sum (not a simple
average) can tighten the joint pose estimate well beyond what one sensor's
number suggests. Worth running the actual CRB sweep before concluding the
margin is a real problem -- it may already be adequately solved by placement
density rather than needing new hardware.

## Sensor comparison table

| Part | Type | Full-scale range | RMS noise floor | Resolution | Package | Verdict for this application |
|---|---|---|---|---|---|---|
| **MLX90393** (current placeholder) | Hall | 5-50mT (progr. gain) | ~0.6-1.3uT @10ms conv | 0.161-3.22 uT/LSB | 3x3mm QFN | Handles near-field fine; far-end SNR ~16:1, tight but workable |
| **MLX90394** | Hall | +/-5mT or +/-50mT (2 fixed ranges) | not fully published in search; same family as MLX90393 | 0.15uT/LSB (5mT range), 1.5uT/LSB (50mT range) | small 6-pin | Similar tradeoff to MLX90393; finer LSB at 50mT range is a plus if noise is comparable -- worth pulling the full datasheet noise table before switching |
| **TLV493D-A1B6** (Infineon) | Hall | +/-130mT | not specified in search results | ~63uT/LSB (12-bit over 130mT) | 6-pin small package | Huge headroom (130mT >> our 11.4mT max, zero saturation risk anywhere in-range) but resolution is coarse -- our far-field 15.7uT signal is <1 LSB. Worse than MLX90393 at the far end despite better near-field margin. |
| **AK09940A** (AKM) | TMR | ~+/-1.2 to +/-2mT (sources vary) | 40-120nT RMS (sources vary) | 0.6uT/LSB (or 10nT/LSB cited elsewhere -- conflicting, verify in full datasheet) | 1.6x1.6x0.6mm | Excellent noise, but range is ~5-10x too small -- would saturate well before the robot even reaches mid-height, let alone 3mm |
| **Bosch BMM350** | TMR | +/-2mT | 190nT (XY) / 450nT (Z) | 16-bit | very small | Same problem: 2mT range vs our 11.4mT near-field -- saturates |
| **MEMSIC MMC5983MA** | AMR | +/-0.8mT | ~40nT RMS | 6.25nT/LSB (18-bit) | small QFN | Best noise/resolution of the bunch, but narrowest range by far -- saturates almost immediately as the robot approaches the plane |
| **TI DRV425** | Fluxgate | +/-2mT (extendable via external feedback resistor, at a sensitivity cost) | 17nT RMS (0.1-10Hz), 1.5nT/sqrt(Hz) @1kHz | very high | needs external compensation coil, single-axis (x3 for 3D) | Best raw noise of anything found, but single-axis, bigger board footprint (external coil), and base range still saturates near-field without external range-extension tuning |

Sources:
- [MLX90393 Datasheet Rev 012 (Melexis)](https://www.melexis.com/-/media/files/documents/datasheets/mlx90393-datasheet-melexis.pdf)
- [MLX90394 product page (Melexis)](https://www.melexis.com/en/product/mlx90394/triaxis-valueoptimized-magnetometer-micropower-low-noise)
- [MLX90394 Datasheet PDF (Melexis)](https://www.melexis.com/-/media/files/documents/datasheets/mlx90394-datasheet-melexis.pdf)
- [TLV493D-A1B6 Datasheet (Infineon)](https://www.infineon.com/dgdl/Infineon-TLV493D-A1B6-DataSheet-v01_10-EN.pdf)
- [AK09940A Product Brief / Datasheet (AKM)](https://www.akm.com/content/dam/documents/products/tri-axis-magnetic-sensor/ak09940a/ak09940a-en-datasheet-myakm.pdf)
- [TDK/AKM joint TMR magnetometer press release](https://www.tdk.com/en/news_center/press/20180109_01.html)
- [BMM350 Datasheet (Bosch Sensortec)](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmm350-ds001.pdf)
- [MMC5983MA Datasheet (MEMSIC)](https://www.memsic.com/Public/Uploads/uploadfile/files/20220119/MMC5983MADatasheetRevA.pdf)
- [DRV425 Datasheet (Texas Instruments)](https://www.ti.com/lit/ds/symlink/drv425.pdf)
- [NVE TMR noise application note](https://www.nve.com/Downloads/SB-00-101_Noise-in-TMR-NVE-Magnetometers.pdf)

## Recommendation on sensor choice

Stick with the MLX90393-class part (or check MLX90394's full noise table --
same family, possibly a cleaner LSB at the 50mT range). Nothing else found
here beats it *for this specific dynamic-range requirement*; the low-noise
TMR/AMR/fluxgate options are built for earth-field-scale sensing (uT range)
and would saturate against our own robot's near-field before ever getting to
use their noise advantage. If the far-end margin still isn't enough after
running the real CRB sweep, the more promising v1.1 levers are:
- **Per-sensor adaptive gain** (MLX90393 supports on-the-fly `GAIN_SEL`
  reprogramming) -- run wide range/coarse resolution when the magnet's
  known to be close, switch to narrow range/fine resolution when it's far.
- **Mixed-gain array**: some sensors fixed at high-range (near-field
  fallback), others at high-resolution (far-field sensitivity), rather than
  one global setting.
- **Longer conversion time / lower output data rate** when the robot is
  known to be moving slowly or is being localized in a static dwell --
  directly buys lower noise per the noise-vs-conversion-time tradeoff already
  documented in `hardware_notes.md`.

## Stronger magnet: what it would actually cost you

A bigger/stronger magnet raises near- and far-field signal together, which
looks like a clean fix for the far-end margin. But this magnet isn't a free
accessory -- per the Nature paper (Fig. 6a, main text), the *same* embedded
N50 cylinders serve **two roles simultaneously**: they provide the net
out-of-plane magnetization used for locomotion torque, AND they "ensure the
structural integrity of the assembly by holding individual units together"
(i.e. adjacent cells snap together magnetically). Changing the magnet changes
both roles at once, not just the one you care about:

1. **Range headroom, not free**: field scales with magnet volume, so a
   stronger/bigger magnet increases the *near*-field too -- at 3mm it's
   already at ~11.4mT against likely a 25-50mT sensor saturation ceiling
   (depending on standard vs thin-IMC part). Doubling the moment roughly
   doubles near-field to ~23mT, cutting that margin in half. You're not
   adding headroom for free, you're shifting the saturation risk from the far
   end toward the near end -- it only nets out well if you *also* upsize the
   sensor's range ceiling to match (e.g. a TLV493D-class 130mT part), which
   then reintroduces the resolution problem from the table above.

2. **Grade alone barely moves the needle**: N50 -> N52 only buys ~3-5% more
   remanence (Br 1.40-1.45T vs 1.43-1.48T per standard magnet-grade specs).
   Real gains require a *bigger* magnet (volume), not a better grade.

3. **Mass and mechanical coupling**: the current magnet is tiny (~2mm
   diameter x 0.5mm, ~12mg by rough estimate) relative to the whole cell.
   Scaling geometry up (e.g. 2x linear -> 8x volume/mass) is probably still a
   small mass fraction of the assembly, but the *structural snap-fit force*
   between magnets is not quantified in either paper -- it's mentioned
   narratively, not given a target Newton value. Increasing magnet strength
   changes that snap force by the same factor, which risks interfering with
   the multistable fold/deploy energy-barrier behavior the paper spent
   significant effort calibrating (bar-and-hinge surrogate model, SI S5) --
   even though that specific calibration is about joint stiffness in the
   *mechanical* rod-based cells, not the magnets directly, a stronger magnetic
   snap force is still a new, unquantified force acting on the same
   fold/deploy transitions. This is a real mechanical-compatibility question
   that isn't answerable from the literature alone -- it would need physical
   testing.

4. **Actuation retuning**: actuation torque tau = m x b scales linearly with
   moment m. A stronger magnet gets more torque per unit of coil field, which
   could be a genuine win (faster/more responsive actuation) -- but the
   coil-current control loop, rise times, and any existing motion tuning were
   calibrated against the current magnet's moment, so this isn't a drop-in
   change either.

**Net take**: a stronger magnet trades one saturation problem (far-end noise
margin) for a different, less-quantified one (near-end saturation +
mechanical snap-fit force + actuation retuning), touching parts of the
system that were carefully calibrated in the original papers. The sensor/gain
and array-density levers above are more contained changes -- I'd exhaust
those (especially just running the CRB sweep to see if multi-sensor fusion
already closes the gap) before considering a magnet change.
