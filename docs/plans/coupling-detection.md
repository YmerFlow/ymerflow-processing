# Detecting coupled data

**Status:** planned, not started. Literature reviewed 2026-10-15.

## Decisions

| | |
|---|---|
| **Detect and cull, never repair** | coupled data cannot be corrected; every source agrees (Appendix B) |
| **One step, one table** | the detector always writes its own per-gate table; `action: flag \| cull` decides only whether it reaches the mask |
| **A QA/QC tool** | unattended first pass, hand-adjusted second |
| **No shapefile required** | a layer raises confidence where it exists; nothing may depend on one |
| **Witnesses corroborate, they do not trigger** | except where geology can imitate the pattern, when corroboration becomes a precondition |
| **Z component only, to start** | the horizontal component and the sweep-across-gates ordering are later work |
| **In scope** | SkyTEM, VTEM, HeliTEM, NRG |

## The gap

`blossPowerlineContaminationTimedomain2017` Fig. 1-1 separates two signatures:

| | looks like |
|---|---|
| "capacitive" | oscillatory behavior at late times |
| inductive / galvanic | **smoothly elevated** dB/dt |

`cull_negatives`, `cull_by_curvature` and `cull_by_decay_rate` are shape tests
on one sounding at a time. They catch the oscillatory kind, because oscillation
breaks monotonicity and curvature. **They cannot catch smooth elevation** —
nothing about its shape is wrong.

Neither can the published automation. Asif state the failure in our terms:

> the impact of this coupling appears as an **enhanced signal level, which could
> easily be interpreted as a geological change** … The automated approach cannot
> distinguish this type of coupling from uncoupled data.

Nor does inversion misfit. A smooth elevation is fitted *comfortably* by a 1D
model, as a conductor that is not there, so **the dangerous case has low
misfit** and anything keyed on misfit selects against it. That includes
`cull_by_external_evidence`'s `misfit` criterion and `data_misfit` (qc_11): a
quiet misfit map on pass one is not evidence the data was clean.

What is available instead is the profile:

> When looking at one sounding (dB/dt vs. time) by itself, **it is nearly
> impossible to tell that the data are affected**; however, when looking at a
> profile of data (dB/dt vs. distance) the effect is quite prominent as a
> **symmetric double peak signature, extending up 300 m away from the powerline,
> and with soundings directly above the powerline largely undistorted.**
> — `blossPowerlineContaminationAirborne2016`

A per-sounding detector run across that pattern flags the flanks and **keeps the
centre**, leaving an island of apparently good data over the conductor.
Detecting the pattern rather than the sounding is the contribution here.

## Scope and limits

These are applicability conditions, not defects. Every filter in this library is
selective — `cull_by_uncertainty` ships no default for `max_uncertainty` because
"what is tolerable depends on the inversion" — and a chain is composed per
survey. Where this step does not suit, it is left out; the docstring says so.

**1. Oscillatory coupling is out of scope.** A late-time sign reversal means
coupling, IP, *or* noise, and IP is ordinary geology, not an exotic case. The
reading is ambiguous and one reading is signal. Appendix C.

**2. IP is culled anyway by `cull_negatives`, and that is acknowledged.** IP
cannot be modeled yet, and data the forward model cannot represent cannot be
inverted correctly. Only the *account* needs changing, in `filter_32`'s
docstring. The same applies to 1D: a known flaw and still industry standard.

**3. A 1D reconstruction basis flags genuine 3D geology.** The criterion tests
one-dimensionality, not cleanliness. Asif:

> If there are 2-D or 3-D effects in the data that cannot be represented by any
> 1-D resistivity model, a low confidence value may be obtained for such data,
> which consequently might be identified as an anomaly by our expert system.

They scope their work out of the harder case explicitly — "designed in a
hydro-geological context. If the infrastructure that induces coupling response
in the data is considered as exploration targets, a different strategy may be
required."

**4. The double peak is not diagnostic.** A steeply dipping conductive sheet is
a current loop in a vertical plane, so the flux-cancellation geometry that gives
the central trough (Appendix A) applies equally to a graphitic fault, a shear
zone or tilted BIF. Off-line conductors channel current into themselves besides,
so the body need not sit under the flight line. Both criteria share this
false-positive class — conductive, compact, strongly responding bodies, which is
to say exploration targets.

What separates a power line from a dipping conductor:

| | discriminates? |
|---|---|
| symmetric double peak with central null | **no** — a vertical sheet gives the same |
| strict symmetry | **partly** — a *tilted* body is asymmetric; a vertical fault is not |
| peak separation vs flight height | **weakly** — shallow narrow, deep broad |
| ruler-straight over kilometres across many lines | **somewhat** — geology is linear but bends and splays |
| the sweep-across-gates ordering | **unknown** — untested |
| an infrastructure layer | **yes** |
| the powerline monitor | **yes, where it responds** |
| a height excursion over the crossing | **yes, where the pilot climbed** |

Everything reliable there is **non-geophysical**. So: the `corroborate` mode is
set per survey; the docstring states where the step suits; and the second pass
earns its keep most where conductive targets are expected.

**Two priors the detector must not carry.** Resistive terrain is not safe —
loop current is limited by contact resistance, not by the background, so there
is no prior about which surveys will be affected (Appendix A). And neither
moment is privileged — the effect peaks in a *time window* near the LM/HM
crossover, so run on both.

## The systems

| | SkyTEM | VTEM, HeliTEM, NRG |
|---|---|---|
| moments | dual — LM and HM | **single**, all three |
| high-rate delivery | yes, 10 Hz | **sometimes** — 10 Hz on VTEM, not guaranteed |
| per-gate error estimates | yes | **no, never** |

**Single-moment is the normal case**, so the LM/HM question is a SkyTEM question
only, and the network count — if the auto-encoder is ever built — is one per
system rather than two.

**"Raw" is not raw.** Every vendor applies an undocumented processing layer
between instrument and delivery. The high-rate product is *minimally processed*.

**The modeled denominator is the default path, not a fallback.** Asif's ε is a
residual in sigmas, and the sigma comes from raw stack scatter. VTEM-like
systems ship no uncertainties at all, and SkyTEM's are the *vendor's*, over the
vendor's stacking window. So `estimate_noise_model` → `assign_uncertainty`
supplies the denominator — weaker, since a modeled sigma cannot see a stack that
happened to be noisy, but the same denominator across systems, so a calibration
is not silently re-scaled when a vendor changes their stacking. Where SkyTEM
uncertainties exist, comparing them against the modeled ones is itself a QC
check.

**The black box is a systematic.** Vendor pre-processing shifts every delivered
sounding away from a clean forward in the same direction, so a network trained
on forwards reconstructs everything slightly badly and the confidence metric
sits uniformly low. Survivable — it moves the threshold, not the ranking — but
it makes `max_reconstruction_error` and `min_confidence` a **per vendor and
system** calibration.

**Which is the strongest argument for the PCA basis.** A linear basis fitted on
the delivered survey learns what delivered transients look like, black box
included, so the vendor's processing lands in the basis rather than in the
residual. Its mirror-image failure must be designed around: a basis fitted on
data containing coupling will absorb the coupling if there is enough of it to
reach the leading components. Mitigations in order of preference — keep the
component count low enough to capture only the smooth decay; fit on segments
with no infrastructure in the search window; or iterate, fitting, detecting and
refitting with detections excluded.

## Workflow: two passes

**Pass one is unattended.** Data arrives, runs the chain, inverts, and produces
a model and a misfit. Nobody adjudicates individual soundings.

**Pass two is guided by pass one.** A human adds culled data back, leaves it, or
culls more by hand. `apply_manual_edits` (filter_33) is the checkpoint;
`version_diff` and `parameter_sweep` show what changed.

### Pass one culls

The alternative — flag and leave it in — is wrong here for three reasons.

**The model is the QC instrument.** A transient cannot be pronounced good by
looking at it; that judgement only exists downstream, as whether the data
inverted into something sensible. Leaving coupling in to be safe corrupts the
very thing pass one exists to produce.

**Coupling does not stay where it was left.** The lateral constraints and the
regularisation spread it — false conductors "get smeared out into the model
domain" — so a flag on three soundings does not contain an artifact thirty wide.
Convergence suffers too; Asif measured the other side, automatically processed
data taking "fewer iterations to converge."

**The cull is reversible.** Pass two reruns from source regardless, so a
pass-one cull costs nothing permanent. The irreversible failure is nobody
noticing it happened — a reporting problem.

So the obligation is to **cull legibly**: the confidence written as a column,
the coupling cull distinguishable in provenance, and the step broken out in
`culled_fraction` (qc_12). With those, over-culling is visible while
under-culling is the one that quietly smears — so the defaults are Asif's
calibrated values, with `corroborate` as the guard where targets are expected.

## The step

### One table, and a switch that applies it

The detector writes **one table of its own**, the same shape as the mask, every
run. All the work lands there. `action` decides only whether it is then applied:

| `action` | writes the coupling table | writes `inuse_{channel}` |
|---|---|---|
| `flag` | yes | no |
| `cull` | yes | yes |

One code path, so nothing can drift between a flagging run and a culling run,
and the table survives both — which is what pass two needs to reverse anything.
This respects the library's rule ("QC writes attributes. Culling writes the
mask."): the table is an attribute; only `cull` touches the mask.

**Not a per-sounding `qs_` verdict**, because coupling affects a band of gates
and the band moves with distance from the line. Every QC check so far judges
auxiliary data — altimeter, tilt, speed — which describes a whole sounding.
This does not.

### The YAML

```yaml
- cull_by_coupling:
    action: flag                      # flag | cull
    criterion: reconstruction         # reconstruction | profile | trend | both
    max_reconstruction_error: 2.5     # sigmas; Asif's alpha
    min_confidence: 0.99              # Asif's beta_max
    window: 7                         # soundings smoothed over; Asif's W
    max_uncertainty: 0.2              # mask noise-dominated gates; Asif's delta_max
    confidence_column: qs_coupling    # per-sounding C_avg, for rollups and re-thresholding
    corroborate: none                 # none | plni | infrastructure | any | all
    infrastructure: null              # optional vector geometry, as filter_30 takes it
    trend_baseline: null              # map distance for the collinearity fit; null = whole survey
    spread_earlier: 1
    disable_tail: true
```

**On the naming.** `max_reconstruction_error` rather than `max_residual`:
`core/misfit.py` already uses "residual" for the inversion's gatewise misfit
("an inversion writes the normalized residual itself as `dbdt_err_ch{n}gt`"),
and `filter_30` takes a `max_misfit` in the same units — standard deviations.
Two different quantities with the same unit and the same name would be read as
one. `min_confidence` rather than `confidence` so the direction is in the name,
matching `max_*` elsewhere.

### Asif's decision machinery

Copied closely, because it is the part that turns an error into a decision:

- mask gates beyond the first where uncertainty exceeds `max_uncertainty` — that
  is noise, not coupling, and not this step's business
- `ε(i) = |d_recon(i) − d_obs(i)| / Δd_obs(i)` — in sigmas
- `C = (1/N) Σ [ε(i) < max_reconstruction_error]` — the fraction reconstructing
- smooth `C` along-line over `2 × window`, so no single sounding decides
- a fuzzy band: clearly high `C` keeps everything, clearly low culls everything,
  and between them only gates with `ε ≤ 1` survive

### The strictness dials

Asif (Section V): "the thresholds α and β_max give an intuition about the
leniency or strictness in the automated data processing."

| parameter | theirs | heavier hand |
|---|---|---|
| `max_reconstruction_error` (α) | 2.5 | **lower** |
| `min_confidence` (β_max) | 0.99 | **higher** |
| `window` (W) | 7 | neither — see below |
| `max_uncertainty` (Δ_max) | 20% | **lower** |

- **`window` is not a strictness dial.** "A larger value of W would smear the
  effect of couplings in the data." It is set by the sounding spacing.
- **`max_uncertainty` is self-correcting upward.** Admitting noisy gates is
  safe, because "the values of ε for such data points will still be low due to
  their higher data uncertainty." Tightening is the sharper move.
- **They are not twitchy** — "not very sensitive and can be adjusted in
  different ways to obtain similar data processing results" — with one caveat
  that is ours: "may not be true if the signal levels are drastically low, e.g.,
  in a very resistive case."

Defaults stay Asif's, as the only values with evidence behind them, with a
docstring note that they were calibrated against *measured* stack scatter and
that a modeled sigma moves them.

### Where it sits in the chain

`docs/reference/llms.txt` keeps an Ordering section — "a wrong order here does
not fail. It produces plausible nonsense." Two placements are forced:

- **after `assign_uncertainty`** — ε divides by Δd_obs; before that there is
  nothing to divide by;
- **before `average_along_line`** — the reason `cull_negatives` is ("or a bad
  gate is smeared into its neighbours"), sharpened by Asif's warning about
  smearing the coupling signature, and quantified by the 10:1 stack measured
  below.

So `estimate_noise_model` → `assign_uncertainty` → `cull_by_coupling` →
`average_along_line`, with `apply_manual_edits` last.

### Corroboration

```yaml
corroborate: none    # none | plni | infrastructure | any | all
```

| mode | cull when the pattern is found and… | |
|---|---|---|
| `none` | always | the default and the design target |
| `plni` | the monitor agrees | |
| `infrastructure` | a mapped feature is within range | |
| `any` | either agrees | patchy layer *and* insensitive monitor |
| `all` | both agree | strictest; protects targets hardest |

`any` and `all` are different tools, which is why "PLNI and shapefile" needed
splitting.

**The PLNI is not one thing**, and which it is decides whether `plni` is sound:

| | tuned coil | bandpass of a late gate |
|---|---|---|
| independence | a separate sensor | derived from the data being judged |
| sensitivity | optimized for the frequency | works *against* the system's own 50/60 Hz rejection |

With a tuned coil, `corroborate: plni` is reasonable. With a bandpass-derived
channel it can block legitimate culls, and `infrastructure` is the better
witness. Record which a survey carries — it is a property of the delivery.

And in both cases the PLNI senses the line being **energized**, not the line
**coupling to our transmitter**. A de-energized line still couples; a loaded
line with corroded grounding emits strongly and couples weakly. So it is read
**relatively**, against the line's own local baseline, never against an absolute
threshold.

**`qc_05_powerline` has already solved that baseline** — it finds local peaks
against a rolling along-line window, because "the ambient level moves within a
line." The corroboration check consumes its excess column rather than deriving
a second baseline.

**The infrastructure layer needs almost nothing new.** `filter_30` already takes
vector geometry as a type and already widens its exclusion with gate time in
`log t`, per moment. Reuse that distance logic; the difference is only that
filter_30 culls on proximity while this queries it.

**A height excursion** is the third witness — `height_above_ground` (qc_01)
computes it, pilots climb over power lines. Weakest of the three, so advisory
only: not every crossing gets a climb, and terrain produces climbs of its own.

### Why this is not `cull_by_external_evidence`

filter_30 is **purely geometric** — it culls everything within a distance of a
mapped feature, having looked at no data. That over-culls quiet crossings (Kang
predicted 174 m and realised ~104 m, because grounding decides and varies pole
to pole) and under-culls loud ones.

| | the map is |
|---|---|
| `cull_by_external_evidence` | an **instruction** — cull within `D`, regardless of data |
| `cull_by_coupling` | **evidence** — the data decides, the map raises confidence |

So they are not competitors and must not be merged. filter_30 keeps its place as
the cheap, deterministic option and for zones that must go whatever the data
says. This step is preferable even where a good layer exists, with
`corroborate: infrastructure`, because it culls what actually coupled.

## The profile criterion

The reconstruction criterion is specified by Asif. This one has nothing to copy.
Scoped deliberately small: **Z only, no template matching, no simulation needed
to start.**

### What sampling allows depends on the product

Measured on demo line 300901:

| product | rate | soundings | spacing |
|---|---|---|---|
| as delivered | **10 Hz** (dt = 0.100 s) | 8995 | **2.23 m** |
| Workbench inversion input | **1 Hz** (478/530 intervals exactly 1.000 s) | 531 | ~36 m |

Against Kang's distribution line — null at 0, peaks at ±30 m, 3% by ±174 m:

- at **1 Hz**, two or three samples across the double-peak core, and a half-
  interval offset means neither null nor peak is ever sampled;
- at **10 Hz**, about **27 points** across the same core.

> **Symmetry survives sampling at any rate. Structure only survives on the
> high-rate product.**

So: run on the delivered product, before any averaging. But build on symmetry
regardless, because the delivered rate is not guaranteed. Shape matching is an
enhancement where the rate allows, never the basis.

### 1. Locate the crossing by its axis of symmetry

**Do not search for the trough** — its position is quantised by the sounding
interval and may fall between samples. Instead cross-correlate the profile
segment against its own reverse; the lag maximising correlation is twice the
offset of the symmetry axis. This uses the well-sampled **flanks** to locate a
centre that may never be sampled, to sub-sample precision.

- **Altitude-invariant.** Height changes the pattern's *width*, not its centre.
  Only positions are compared across lines, never widths.
- **Angle-invariant.** A straight path crossing a straight line is symmetric at
  any angle; only apparent width stretches, by `1 / sin θ`.
- **Yields its own uncertainty**, from the sharpness of the correlation peak.

Each candidate carries a position, a measured width, an amplitude and a position
uncertainty.

### 2. Width is measured, never predicted

Three reasons converge: contact resistance is unknowable and governs the
realised extent; line class is often unknown and loop area spans an order of
magnitude; crossing angle is unknown and stretches width by `1 / sin θ`. The
derived extent from Appendix A bounds the *search window* and nothing more.

### 3. Collinearity across every line, regardless of bearing

Candidates from all lines go into one map-coordinate set; the test is for
collinear subsets with **small residual scatter** — a Hough transform or RANSAC.

- **No flight-line / tie-line distinction.** The test works on `(x, y)` and has
  no reason to know a line's role. Tie lines are not assumed perpendicular and
  are not required; differing bearings simply improve the fit geometry.
- **Scatter, not collinearity, is the test.** A transmission line is straight to
  within metres over 10 km; geology bends and splays. That is a number with a
  threshold, which is what makes it a discriminator.
- **Missing lines are harmless.** Hough and RANSAC tolerate gaps, and a
  non-detection on a line flown high is *explicable*, since qc_01 has the height.
- **`trend_baseline` sets the fit length.** A guardrail is straight over 200 m
  even where it wanders over 2 km. Long baseline with tight scatter finds
  transmission and rail; short with loose finds guardrails and road-following
  distribution.

**This is what lets the per-line pass be permissive.** Tune it for *recall*, let
the trend stage supply *precision* — which removes the need for a precise
per-line detector, the hard part.

**Sort, not reject.** A candidate that fails collinearity is not thereby noise;
an isolated cultural source is non-collinear by nature. RANSAC discards
non-collinear points by construction, so the trend stage must **classify**:

| | verdict |
|---|---|
| collinear over its baseline | confident; infrastructure; safe to `cull` |
| isolated | uncertain; `flag` for review regardless of mode |

**What it gives back:** a bearing per trend, hence `θ` per line and a
perpendicular extent; candidate discovery with no infrastructure layer; and
orientation independence.

**Its one structural cost:** it is **survey-level**, unlike every other filter
here. If a survey is processed line-by-line it cannot work — so it must detect
that it has one line and say so.

### 4. Lines parallel to the flight direction — unsolved

If the power line runs *along* the flight line there is no uncoupled background,
nothing stands out, and the detector reports clean with no sign that it failed.
Not an edge case: lines follow roads and valleys, and grids are laid out against
the same terrain.

Partial answers: lines of a different bearing in the same survey still cross it
and the trend test consumes them equally; an infrastructure layer gives the
bearing directly; and **at minimum the detector must refuse to claim a clean
verdict** where a trend or mapped feature runs within some angle of the flight
direction. A hole that announces itself is survivable; a confident pass is not.

### 5. Four regimes, because it is not always a power line

| regime | examples | profile | collinearity | reconstruction |
|---|---|---|---|---|
| **resolved, straight** | transmission, railway | yes | yes | yes |
| **resolved, curving** | guardrail, debris fencing, distribution on roads, farm fencing | yes | no | yes |
| **areal / unresolved** | vineyards, towns, landfills, industrial sites | **no background** | no | fires everywhere |
| **point** | wreckage, well casings, tanks, silos, pivots | **too few samples** | **rejects it** | **the right tool** |

**Railways are the best case** — straighter than power lines, continuous welded
steel, bonded for signalling.

**Areal sources: edge-detect then fill.** Vineyard rows at 2–3 m against 30 m
soundings are far past Nyquist, so the rows never resolve and what is seen is a
plateau with a step at the block boundary. Find the entry and exit steps and
mark between them. Simpler than the symmetric search, and it partially rescues
§4, because a region has edges even where a line inside it has no background.

**Point sources are the limit of the approach.** A compact conductor at 40 m has
a 50–100 m footprint — two to four soundings at 30 m, too few for the symmetry
estimator. The reconstruction criterion handles them well, which is the clearest
justification for carrying both. But an isolated cultural conductor and an
isolated mineral target are *the same measurement*, and there is probably no
geophysical answer. Three weak discriminators, none decisive:

- **Time constant.** τ ≈ μ₀σa²/π². Metal at σ ≈ 10⁷ S/m gives *seconds* —
  effectively non-decaying across the gate window — against a few milliseconds
  for a sulphide body at σ ≈ 10 S/m and 50 m scale. **An anomaly that does not
  decay at all is metal.** Arithmetic to be checked, not trusted.
- **Shallowness.** Wreckage is at surface and hits early gates hardest.
- **Spatial coincidence** on adjacent lines, where spacing is fine enough
  relative to the footprint — at 100–500 m it usually is not.

**Debris-fall fencing is unsolved**, not parameterised. Rockfall *mesh* is a 2D
conductive sheet draped on a slope, so it presents as an inclined conductive
sheet — the dipping-conductor mimic — it follows contours so it curves, and it
lives where terrain is steepest and the 1/r⁶ amplitude swing is largest.

### Later, once the shed stands

**The horizontal component.** Deliveries carry one (`HM_X_G01`) and the library
keeps it as an *unclaimed* array. **Do not declare it in the GEX** to make it
available: `DualMomentTEMXYZSystem` hardcodes `rx_orientation='z'` and would
read a declared X channel as Z — a confident wrong model, with a test asserting
exactly that. Read it through `unclaimed_arrays`. Optional and degrading loudly,
because single-component receivers exist.

What it may buy is a sharper position estimate. The response goes as (Tx→loop) ×
(loop→Rx), both antisymmetric about the crossing, whose product is symmetric and
positive — which is why the Z anomaly is a symmetric positive double peak rather
than a sign reversal. If so the central null is partly an *excitation* null, no
receiver orientation fills it, and X is instead antisymmetric with a sharp
**zero crossing** at the line. Zero crossings localise better than broad
troughs. It also predicts concentric (VTEM) and offset (SkyTEM) systems differ,
their two nulls coinciding in one case and separating in the other.

**Derived, not read. Check before building on it** — the SimPEG run queued for
the dipping-plate test settles it.

**The sweep-across-gates ordering** — late gates lighting first on approach,
going out last on departure. Still the only candidate discriminator that is
geophysical rather than circumstantial, and still untested.

## Implementation

### Layout

```
ymerflow_processing/core/
  cull/filter_NN_coupling.py     the step: parses params, orchestrates, reports
  coupling/
    __init__.py
    table.py                     the per-gate table: naming, ensure, apply
    reconstruct.py               Reconstructor protocol + PCAReconstructor
    reconstruction.py            Asif's criterion over a Reconstructor
    profile.py                   per-line symmetric-departure candidates
    trend.py                     cross-line collinearity, classification
    corroborate.py               plni / infrastructure / height witnesses
```

A package rather than one long module, because the criteria are independently
testable and the reconstructor must be swappable. Each file stays small; no
class hierarchy.

**Step number** is unassigned. The constraint: it must sort among the culling
filters and below 33, since `apply_manual_edits` replays last. 26, 28 and 34 are
the candidates; 28 sits naturally between curvature (27) and noise floor (29).

### The table

The fiddliest part, because it is a new array family and the library is strict
about those.

```python
# core/coupling/table.py
PREFIX = "coupled_"          # a PREFIX, not a suffix
STEP_CODE = NN

def key_for(inuse_key: str) -> str: ...
def ensure(dataset, channel) -> pd.DataFrame: ...   # creates zeros if absent
def apply(dataset, channel, table, step_code) -> dict: ...
```

Requirements, each with a test:

- **Prefixed, never suffixed.** Channel naming patterns are anchored at the
  start, so `dbdt_inuse_ch1gt_coupled` would still match the in-use pattern and
  discovery would find two masks for one channel. This is the lesson `why_`
  already encodes.
- **Not reported as unclaimed.** `unclaimed_arrays()` and `inventory.take()`
  must agree about it — there is already a test asserting those two agree, and
  a new array family that only one of them knows about will break it.
- **int8**, mirroring the provenance encoding: `0` clean, `+1` detected
  directly, `-1` swept up by the culling policy. `cull()` already draws that
  distinction with `+code` / `-code`, so the table and the provenance speak the
  same language.
- **Post-policy contents** — what *would* be culled, since the point of flagging
  is to show what the cull would do.
- **Survives export**, and round-trips through the msgpack writer.

Applying it is one call, because the mask API was built for this:

```python
cull(dataset, channel, bad=table != 0, step_code=STEP_CODE, direct=table > 0)
```

### The reconstructor interface

```python
# core/coupling/reconstruct.py
class Reconstructor(Protocol):
    def fit(self, transients: np.ndarray) -> None: ...
    def reconstruct(self, transients: np.ndarray) -> np.ndarray: ...

class PCAReconstructor:
    """SVD over the gate dimension, in log space, fitted on this survey."""
    def __init__(self, n_components: int = 4, log_space: bool = True): ...
```

`fit` then `reconstruct` is all the criterion needs, so the auto-encoder drops
in later without the step changing. `n_components` is the knob that controls the
absorb-the-coupling failure: too many and the basis learns the coupling too.

### The criteria

```python
# core/coupling/reconstruction.py
def detect(dataset, channel, reconstructor, *, max_reconstruction_error,
           min_confidence, window, max_uncertainty) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per-gate bool, per-sounding C_avg)."""

# core/coupling/profile.py
@dataclass
class Candidate:
    line: str; x: float; y: float
    width: float; amplitude: float; position_sigma: float

def candidates(dataset, channel, *, search_window, min_prominence) -> list[Candidate]:
    """Per line: symmetric departures from the along-line background."""

# core/coupling/trend.py
@dataclass
class Trend:
    bearing: float; scatter: float; members: list[int]

def fit(candidates, *, baseline, max_scatter) -> list[Trend]:
    """Hough/RANSAC over map coordinates. Baseline None = whole survey."""

def classify(candidates, trends) -> list[str]:
    """'collinear' | 'isolated' — sort, never reject."""
```

### How much of this already exists

Asif's method maps almost one-to-one onto parts already built, which is why
stage 1 is short:

| the method needs | exists as |
|---|---|
| per-gate uncertainty, ε's denominator | `estimate_noise_model` → `assign_uncertainty` |
| mask gates past the first above Δ_max | `cull_by_uncertainty(max_uncertainty, …)` — same parameter, same meaning |
| widen earlier, disable tail | `spread_earlier`, `disable_tail` on every culling filter |
| subtractive culling with per-datum provenance | `mask.cull(…, step_code, direct=…)` |
| what was culled, reported | `culled_fraction` (qc_12) |
| a before/after figure | `figures.CULLING` |

What is genuinely new is small: a reconstruction basis, and about fifty lines of
arithmetic the paper fully specifies. **The time goes into integration** — the
table is a new array family and this library is strict about those — not into
the detector.

### Work sequence

1. **The table.** Naming, `ensure`, `apply`, export round-trip, and the
   discovery/inventory tests. Nothing else can be tested until this exists.
2. **The step skeleton.** Registration in `setup.py`, `action` switch, parameter
   validation, `StepReport`, a `figures.SHAPES` entry (`CULLING`), and a
   regenerated `docs/reference/`. Both enforced by tests that fail rather than
   warn.
3. **`PCAReconstructor` + the reconstruction criterion.** Asif's arithmetic;
   roughly fifty lines once the table exists.
4. **Run it on demo line 300901** against the Workbench `InUse` mask. Calibrate
   `max_reconstruction_error` and `min_confidence` with `parameter_sweep`.
5. **The profile criterion**, per line, with the symmetry estimator.
6. **The trend criterion**, cross-line, with classification.
7. **Corroboration**, consuming qc_05's excess column and filter_30's distance
   logic.

Steps 1–4 are stage 1 and are close. 5–7 are stage 2 and have no precedent.

### Tests to write

Named for the property, in the library's style:

- a suffixed key would be discovered as a second mask — the trap, asserted
- the table and the provenance agree on direct versus collateral
- `action: flag` leaves `inuse` untouched and the table populated
- `action: cull` produces exactly the mask that applying the table gives
- re-running a flag pass is idempotent
- the trend criterion on a single-line dataset reports that it cannot run
- a sounding with no reconstruction error is not culled
- gates above `max_uncertainty` are excluded before the decision, not after
- a synthetic symmetric departure is located to better than one sounding
- a synthetic *asymmetric* departure is not located as a crossing
- collinear candidates are classified `collinear`; a lone one, `isolated`
- the emitted diff round-trips through `apply_manual_edits`

### Front end

Not required for stage 1, and two small pieces:

- **One more state in `ChannelPlot`.** It already renders a four-state per-gate
  overlay (auto/manual × enabled/disabled) from the mask overlaid with a pending
  edit. The coupling table is a third category of the same kind, fed by an array
  shaped like one it already reads. Two states if direct and collateral are
  drawn differently, which is worth doing.
- **A path to load a pipeline-emitted diff.** The table converts to the existing
  edit format in a few lines — `layer_data["InUse_{channel}"]` with NaN meaning
  no override, `apply_idx` from the affected rows. The frontend currently builds
  that overlay only from its own in-memory reducer. Giving it a load path makes
  every future proposing filter reviewable the same way.

## Validation

### The calibration set already exists

`deps/ymerflow-demo-data` carries a SkyTEM 304 survey — line 300901 and a block
— in two forms.

| file | carries |
|---|---|
| `as_delivered/skytem_as_delivered_line_300901.xyz` | `LM_Z_G01`, `HM_Z_G01`, `HM_X_G01`, `PLNI`, `Height`, `AngleX/Y`, `Curr_LM/HM`, `DEM`, `GdSpeed`, magnetics. 8995 soundings, 10 Hz. **No uncertainties.** |
| `agf_inversion/inversion_input_data_line_300901.xyz` | Workbench export, 2019: `Gate_Ch01_01…28`, `Gate_Ch02_01…37`, `STD_Ch0*`, **`InUse_Ch0*`**, `resdata`, `restotal`. 531 soundings, 1 Hz. |
| `agf_inversion/inversion_forward_response_line_300901.xyz` | the forward response, so misfit is computable |

`InUse_Ch0*` is **a processor's own per-gate culling decision on this line** —
the hand-marked ground truth, already in the repository. Gate counts agree
across the two forms (28 LM, 37 HM), so the gate axes align without
interpolation.

### Two levels of ground truth

Only 531 soundings survive of the 882 seconds the line spans: 52 gaps totalling
~351 soundings, the largest 29 s (about a kilometre). **Forty percent of the
line was removed outright.** So the judgement is recorded twice — whole
soundings deleted, and gates masked on the survivors — and any scoring must
account for both.

The gap structure alone is worth plotting against `PLNI` before any detector
exists: 52 episodes of up to a kilometre is what a map of culture looks like.

### The join is spatial

Fids differ between products and cannot be matched. Join on position —
`E_UTM14N_m` / `N_UTM14N_m` against `UTMX` / `UTMY`, after confirming a shared
datum — many-to-one, roughly seventeen delivered soundings per Workbench
sounding. Aggregate our verdicts up to the Workbench positions rather than
interpolating the mask down.

### The experiment

`InUse` records *that* a gate was culled, not *why* — a processor removes
turn-off gates, noise-floor data, negatives, altitude excursions and couplings
alike. So run the chain twice:

1. the standard chain → mask **A**
2. the same chain **plus** `cull_by_coupling` → mask **B**
3. both scored against `InUse` *and* against the whole-sounding deletions

**B − A is this step's contribution**, against a real processor's judgement, and
it separates the two failure directions: gates the processor kept that we culled
are over-culling; gates it culled that we kept are under-culling.

This also makes Asif's 96% *measurable* — a different expert, survey and a
decimated join, so not the same number, but the same measurement. And the data
was culled because something was wrong with it, whatever the reason recorded, so
agreement with that judgement is a fair test even where the reason is unknown.

Remaining caveats: the mask is aggregate rather than coupling-specific, the join
loses resolution, and it is one line of one survey by one processor.

### Synthetic data, and its circularity

Kang released the notebooks behind the paper and the 3D code is SimPEG-EM,
already in the stack as `YmerFlow/simpeg`. Worth building for controlled sweeps
of flight height, loop count and ground-pathway resistivity.

**But it is partly circular.** The coupling would be generated by a model whose
weakest parameter — contact resistance, which Appendix A shows governs the whole
circuit and which nobody can measure — decides the answer. The precedent is
ours: `blossPowerlineContaminationAirborne2016` built a forward model giving
results "consistent with the collected gTEM data" that "do not reproduce the
amplitude or spatial-scale of powerline distortion observed in ATEM data."
Kang's 3D treatment is better and still assumes negligible contact resistance,
which is why their field data came in at ~60% of prediction.

So synthetic data measures **sensitivity to parameters** and catches gross
failures. It does not establish that the detector finds real coupling.

## Cost

| | |
|---|---|
| soundings, 2700 line-km at 30 m | 90,000 |
| passes, × 2 moments | 180,000 |
| per pass | ~97k MAC |
| total | **≈35 GFLOP** — seconds on CPU |

Asif measured **0.3 ms per raw data stack** on one Xeon Gold 6132 core — about
54 s for the survey above, single-core. **Inference cost is not a consideration
in this design.**

Training, if the auto-encoder is ever built: their 1M resolvable-model dataset
is publicly available, and their network took ~13,000 epochs at ~6 s/epoch ≈
**22 GPU-hours, once**. The recurring cost is one network per system per moment
— a release-time cost, not a per-survey one.

## Staging

0. **The table and the switch.** Worth doing first and almost free: it makes
   every later stage reviewable rather than merely runnable.
1. **The framework with a PCA reconstructor.** Asif's arithmetic, the confidence
   column, smoothing, the fuzzy decision, mask and provenance. No new
   dependency. Fitted on the delivered survey it also absorbs the vendor's
   pre-processing instead of mistaking it for an anomaly.
2. **The profile criterion, Z only**, then the trend criterion. Works on both
   systems and on delivered data, needs no uncertainties — and is **the riskiest
   stage**, being the only part with nothing to copy.
3. **The auto-encoder reconstructor**, behind the interface from (1), once there
   is a trained artifact and a decision about shipping weights in a GPL repo.
   May never pay for itself, given per-system retraining and the 1D blind spot.

## Open questions

**Needing the data**

- **What do `max_reconstruction_error` and `min_confidence` become against a
  modeled sigma?** Asif's 2.5 and 0.99 are tied to measured stack scatter, and
  with vendor processing in the chain the calibration is per vendor and system.
  `parameter_sweep` against line 300901 is the tool.
- **How far does vendor pre-processing move the reconstruction residual?** If
  the offset is uniform it only moves the threshold; if it varies along the
  line, the black box is doing something adaptive and the criterion is shakier
  than it looks.
- **The per-line detector's numbers** — how wide "locally confined" is in
  soundings, how the along-line background is estimated (rolling window as
  qc_05, polynomial as Polzer, or otherwise), and what threshold keeps the pass
  permissive without flooding the trend stage.
- **How is `trend_baseline` chosen** — a fixed map distance, a count of adjacent
  lines, or a sweep reporting what was found at several scales?

**Needing one SimPEG run** — one dipping plate, one power-line loop, same flight
line

- **Does the sweep-across-gates ordering discriminate?** The only geophysical
  candidate left. If it does, the mineral restriction loosens considerably.
- **Does the horizontal component behave as derived above?** Same run.

**Needing a decision**

- **What does "working" mean?** Per gate, per sounding, or per crossing? Per
  crossing is probably the honest unit, since missing a crossing and trimming
  its edges differently are not comparable errors. **Decide before building.**
- **Where is `corroborate` declared?** A project-level fact, not derivable from
  data, so it probably belongs with project configuration rather than buried in
  a step.
- **Does line class need to be an input?** Voltage class proxies for tower
  height and span and would sharpen the window, but makes the criterion depend
  on data we often lack.
- **Where would a trained model live?** Weights are not source, the repository
  is going public under GPL-3.0, and a model artifact carries its own provenance
  and licensing questions.
- **The step number** — 26, 28 or 34.

**Needing someone to ask**

- **Which PLNI derivation does each vendor ship** — tuned coil or bandpass? It
  decides whether `corroborate: plni` is sound or harmful for that delivery, and
  the data does not announce it.

**Needing an arithmetic check**

- **The metal time constant**, τ ≈ μ₀σa²/π², which is the basis for the only
  point-source discriminator offered.

---

# Appendix A — The physics

`kangEffectPowerLines2021` simulates the mechanism in 3D — the wire between
poles, the wire down each pole, the ground electrodes and the ground pathway
closing the circuit — citing `blossPowerlineContaminationTimedomain2017` for the
circuit treatment of contact resistance.

## Why the centre is clean

> When AEM loops are right above the power line, there will be no magnetic flux
> (generated by induced currents flowing in the power-line loop) intersecting
> the receiver loop … As the AEM loops move away from the power line, the amount
> of magnetic flux intersecting the receiver loop increases, reaching a peak when
> the coupling is maximized, and then it decreases as the loops move farther away.

Flux cancellation. The double peak with a null between is **what the geometry
requires** — which is also why geology can produce it, a steeply dipping
conductor being the same geometry.

## The window is not a constant

Kang's modeled line: **10 m poles, 80 m spans**, 15 loops over 1.2 km,
transmitter at 40 m, ~20 Ω·m half-space. Result: peak TPPE ≈ 44% at **~30 m**,
below their 3% noise level beyond **174 m**, with field data showing clean
soundings from only **~60%** of that.

**That is a distribution line.** Loop area is span × pole height, so a
transmission line on 40 m towers at 350 m spans encloses roughly seventeen times
the area — though inductance rises with loop size and limits the current, so the
amplitude does not scale with area (below). `bloss…2017` Fig. 2-6 separates the
classes for this reason: transmission above 230 kV, subtransmission ~100 kV,
distribution below 69 kV, residential under 1000 V.

Kang's figure is an upper limit three ways: **negligible contact resistance** is
assumed (their own field data realised ~104 m against 174 m predicted, which
they attribute to "degradation of the grounded electrodes"); **no catenary**,
and sag reduces loop area more the longer the span; and it is **an 800 m² loop**.

What the detector takes from this is a dependence, not a distance:

| grows with | shrinks with |
|---|---|
| span × pole height (loop area) | transmitter height |
| number of contributing loops, to ~15 | **total loop resistance** = contact resistance (unknowable) |
| | loop inductance, which rises with loop size |
| | catenary sag |

Absent from that list: the bulk resistivity of the ground.

## Contact resistance limits the loop, not the background

Treat the skywire as a circuit (`bloss…2017` §2.2.4.2): the top wire, two pole
wires, the earth return, and two contact resistances. The conductor terms are
trivial; the earth return is easy to get backwards.

Current between two grounded poles spreads into the half-space as a curved
bundle of flow lines — the current banana. From the pole-pole array, the
transfer resistance between two surface electrodes separated by `a` is

```
R_earth = ρ / (2 π a)
```

**inversely** proportional to separation. That is the general result the
electrical and MT literature states plainly: **more volume engaged means more
parallel pathways means lower resistance.** Any intuition built on "longer path,
more resistance" — which a fixed-cross-section model encodes — has the sign
wrong.

A 100 m span on 100 Ω·m gives **0.16 Ω**, against 4.8 mΩ for 150 m of conductor.
The case intuition calls protective — **1000 Ω·m with poles 30–40 m apart — is
still only 5.3 to 4.0 Ω.**

So the thesis: "Neglecting the CR at the grounding points the total resistance
of the Skywire is 0.16 Ω! This low number suggests that the CR is the limiting
factor here." **Contact resistance dominates** — aimed at <20 Ω, degrading
unpredictably with corrosion and season — and it is the one quantity no survey
can measure.

Three consequences:

1. **Do not deprioritise resistive terrain.** Bulk resistivity barely enters,
   and the *percentage* effect moves the other way, since "an increase in
   ρ_background causes a decrease in the amplitude of the subsurface signals,
   resulting in a net increase in the percentage power-line effect." Resistive
   ground may be the harder case.
2. **The threshold that matters is a circuit threshold.** The thesis found one
   directly: "Resistances above 100 ohms yield Skywire contributions so low in
   amplitude that when superimposed on the geologic response the data were
   largely unchanged." That fits Kang's electrode-degradation observation; both
   say the grounding decides.
3. **The snowflake problem, as a circuit.** The quantity deciding whether a
   crossing is loud or silent cannot be known, so carry no prior about which
   surveys will be affected.

Loop inductance tempers the area scaling: **0.67 mH** for a 225 m × 20 m
transmission-style skywire against **0.14 mH** for 100 m × 10 m, and "as
inductance increases, the maximum amplitude of the Skywire response lessens",
with the response pushed later in time.

## The effect peaks in a time window, not a moment

Lower loop resistance lengthens the time constant and pushes the response later;
higher resistance pulls it earlier. Two independent codes agree — Kang in ρ_g
(Figure 9, late-time HM effect growing as ρ_near falls from 340 to 1 Ω·m), and
`bloss…2017` §3.1.2.4 varying loop resistance directly: "As Skywire resistance
increases, the observed anomaly is earlier in time and as Skywire resistance
decreases the observed anomaly is pushed to later times."

Kang's field data puts the peak between the moments:

> (1) the magnitude of the power-line effect in the LM data **increases with
> increasing time channel**, and (2) the power-line effect in the HM data **is
> strong in the early time channels, but it decreases with the increasing time
> channels**.

LM spans 0.016–1.4 ms and HM 0.36–11 ms, so both point at the **overlap window,
roughly 0.4–1.4 ms**. Which moment looks worse depends on where that window
falls among each moment's gates — survey-dependent, so run on both.

**Field LM is quieter than modeled LM**, and Kang give the reason: "the first
few time channels of the LM data show a power-line effect smaller in magnitude
than the simulation results; this is due to the increasing flight height around
the power line." Pilots climb over power lines, and the LM response "shows
greater sensitivity to changes in h than does the HM response." The two-way
dipole path goes as 1/r³ down and 1/r³ back — **1/r⁶** — so a 20 m climb on a
40 m nominal is roughly a factor of 11, larger than most effects discussed here.

**TPPE is a ratio.** Kang's 44% is a percentage of the subsurface response, so a
moment can look worse simply by having a weaker earth signal at the times it
measures. Absolute and percentage comparisons are different questions, and the
one that matters for inversion damage is the percentage.

---

# Appendix B — Why detection, not repair

Couplings add a response **coherent with the earth's**, because the transmitter
induces current in the conductor and the receiver sees the sum.

> Couplings to such infrastructure influence the measured signal to a degree
> that cannot be corrected for, making it necessary to completely remove coupled
> data from the processed dataset. — `andersenArtificialNeuralNetworks2016`

> Couplings to infrastructure cause the data to be corrupted to a degree that
> cannot be corrected. Therefore, the TEM data denoising approaches … are not
> feasible. — `asifAutomatedTransientElectromagnetic2022`

**It is also not a harmonic problem.** `larsenRemovalPowerlineNoise2022` removes
50/60 Hz harmonics from fully-sampled time series and notes TEM already
suppresses those at acquisition, by synchronous detection with alternating
transmitter polarity. Our data arrives stacked; the harmonics are gone before we
see it.

## Stripping has been done, and does not transfer

`polzerStrippingCulturalConductor1990` strips cultural response successfully,
without modeling the grounding:

> we do not have to model the response of a grounded conductor to remove most of
> its effect from the data. This is because most of the vertical magnetic field
> response arises only from the induced currents flowing within the conductor
> itself which need only be estimated by measurement.

At each delay time it decomposes a short traverse into the field of a
unit-ampere current element along the conductor's known path, scaled by an
unknown current, plus a smooth quadratic background, and solves least squares.

Three things stop it transferring: it needs **the 3D path of each conductor**
(skywires being the hard case, since height follows a catenary); it needs
**dedicated traverses** to estimate the current, where a production line crosses
once at whatever angle the survey gives; and because the conductor is grounded
at many points the current is not constant along it, so each segment needs its
own traverse. **A survey-design method, not a processing method.**

The modeling route was tried and did not reach ATEM scale.
`blossPowerlineContaminationAirborne2016` built a 'skywire' RL analogue after
Polzer, with results "consistent with the collected gTEM data" but which "do not
reproduce the amplitude or spatial-scale of powerline distortion observed in
ATEM data." Parameterising it needs the contact resistance at each pole and the
capacitance between ground wire and ground, neither knowable per survey.

**What Polzer gives us anyway:** his decomposition is the ancestor of the
profile criterion. He fits *smooth background + conductor response* to
**subtract**; we look for *smooth background + symmetric departure* to
**detect**. Detection is the weaker ask, and weaker exactly where Polzer is
expensive — no conductor geometry, no template, no dedicated traverse. His
result that the stripped response is "symmetric about the loop with a positive
amplitude envelope … of about 40%" independently corroborates the symmetry.

---

# Appendix C — The oscillatory signature is usually geology

A late-time sign reversal has at least three causes, as the library's own
specification says — `01`, quoted in `filter_32`:

> a sign change indicates coupling, **an IP effect**, or noise dominating

`cull_negatives` culls all three identically, and one of them is geology.

**IP is not a special case.** Any charging interface produces it — a conductive
clay layer in sand or sandstone polarizes at the contact, an ordinary
sedimentary situation. In mineralized ground sulphides are strongly polarizable
and effectively glow. And IP is present long before it is strong enough to drive
the transient negative; a negative is the end of a continuum, not the start.

It is **recoverable information**: `macnaeQuantifyingAirborneInduced2016` and
`macnaeQuantitativeEstimationIntrinsic2016` quantify and estimate intrinsic IP
parameters from airborne EM, and `grombacherInducedPolarizationEffects2021`
documents the effects in real ATEM data.

**"Capacitive" means two different things.** `kangEffectPowerLines2021` uses it
for a buried insulated cable acting as a capacitor, concluding within that scope
that "without large metallic objects such as pipes, it will not be possible to
observe significant capacitive coupling in AEM data." That is about
well-associated noise sources. It does not cover the late-time oscillation we
most often meet, which is polarizable geology and needs no infrastructure.

**Culling IP is deliberate.** IP cannot be modeled yet, and data the forward
model cannot represent cannot be inverted correctly, so culling it is the honest
response and what the industry does. What is worth changing is the *account*:
`filter_32`'s defence of `disable_tail` — "a transient does not recover once it
has gone bad" — is true of coupling and noise and **false of IP**, where the
transient is doing something physical and repeatable. Recording that costs
nothing and matters when an IP-capable forward model arrives.

Asif's inverse limit bounds what a good reconstruction score means: "when a
coupling response mimics a geological response. If such data can be represented
by 1-D model parameters, the autoencoder reconstructs the data with low error
and the data will not be discarded." **The criterion tests one-dimensionality,
not cleanliness.**

---

# Appendix D — Sources

Keys are in `~/Codes/_refs/library.bib`; PDFs in the YmerFlow shared library.

- `asifAutomatedTransientElectromagnetic2022` — the auto-encoder expert system,
  and the most useful paper here. **Section V** carries the four tunable
  parameters, the raw-stack requirement, retraining for other systems, and the
  authors' own statements of the 1D/3D and mineral-exploration limits
- `kangEffectPowerLines2021` — 3D simulation: flux cancellation, the distance
  and TPPE numbers, LM/HM and transmitter-height sensitivities, field validation
  against 41 water wells, and released notebooks
- `blossPowerlineContaminationTimedomain2017` — the signatures, the profile
  characterisation, the skywire circuit (§2.2.4.2) and the resistance and
  inductance experiments (§3.1.2.4)
- `blossPowerlineContaminationAirborne2016` — the profile characterisation and
  the smearing failure
- `andersenArtificialNeuralNetworks2016` — the earlier supervised ANN
- `polzerStrippingCulturalConductor1990` — stripping by measured current
- `xieImprovedPrincipalComponent2014` — PCA denoising; a reconstruction basis
- `larsenRemovalPowerlineNoise2022` — harmonic removal, **not applicable here**,
  but its treatment of remote-reference cancellation is the nearest thing in the
  library to an MT source and describes what the powerline monitor is
- `macnaeQuantifyingAirborneInduced2016`,
  `macnaeQuantitativeEstimationIntrinsic2016` — airborne IP quantified
- `grombacherInducedPolarizationEffects2021` — IP in real ATEM data
- Yin & Wang 2015, powerline ground resistance from helicopter EM. **Not read**

**Not in the library:** no magnetotelluric or self-potential papers. The
volume/pathways result in Appendix A is standard in that literature but is
stated here without a citation.
