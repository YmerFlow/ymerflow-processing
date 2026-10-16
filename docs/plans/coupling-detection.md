# Detecting coupled data

**Status:** planned, not started. Literature reviewed 2026-10-15.

**Decided:**

- **Detect and cull, never repair.** Coupled data cannot be corrected; every
  source agrees. See Appendix A.
- **Pass one culls**, and culls legibly. It does not merely flag.
- **Primarily a QA/QC tool**: unattended first pass, hand-adjusted second.
- **Both VTEM and SkyTEM** are in scope, and the differences matter.
- **The powerline monitor corroborates; it does not trigger** — except where
  geology can imitate the pattern, when corroboration becomes mandatory.

## Summary

Our culling filters are shape tests on one sounding at a time, so they catch
oscillatory coupling and miss **smooth inductive elevation** — a galvanically
coupled sounding is a plausible-looking decay that is simply in the wrong place.
Both published ML approaches miss it too, and Asif say so. That is the gap.

The gap is visible **in profile, not per sounding**: a symmetric double peak with
a central trough, the trough being flux cancellation directly over the line. No
automated method in this library looks for it.

So: one new step, `cull_by_coupling`, with two criteria behind one interface — a
**reconstruction** criterion copied from Asif (residual in sigmas against a
reconstructed transient, with their decision machinery and their four tunable
thresholds) and a **profile** criterion that searches for the pattern. The
reconstructor is swappable, so a linear PCA basis works today with no training
and the auto-encoder drops in later unchanged.

**The hard part is not detection, it is discrimination.** A steeply dipping
conductive sheet produces the same profile signature as a power line, and a 3D
target produces the same reconstruction failure. Nothing geophysical separates
them reliably; the dependable witnesses are an infrastructure layer, the
powerline monitor, and the pilot's climb. That is why this is a QA/QC tool with
a human second pass rather than an unattended cleaner, and why it must not run
unsupervised on mineral exploration data.

## The gap this fills

`blossPowerlineContaminationTimedomain2017` Fig. 1-1 separates two signatures:

| | what it looks like |
|---|---|
| "capacitive" | oscillatory behavior at later times |
| inductive / galvanic | **smoothly elevated** dB/dt curves |

The existing culling filters — `cull_negatives`, `cull_by_curvature`,
`cull_by_decay_rate` — are shape tests on one sounding at a time. They catch the
oscillatory kind, because oscillation breaks monotonicity and curvature.

**They cannot catch smooth elevation.** Nothing about its shape is wrong.

Neither can the published automation. Both ML approaches judge one sounding, or
one raw stack, at a time, and Asif state the failure mode in exactly our terms:

> a buried power line excited by the tTEM transmitter … the impact of this
> coupling appears as an **enhanced signal level, which could easily be
> interpreted as a geological change**. The manual operator has removed all data
> in the vicinity of the road … The automated approach cannot distinguish this
> type of coupling from uncoupled data.

**The state of the art fails on precisely the half our shape filters miss.**

Nor does inversion misfit catch it. A smooth elevation is fitted *comfortably*
by a 1D model, as a conductor that is not there — so the dangerous case has
**low** misfit, and anything keyed on misfit selects against it. See "And its
`misfit` criterion is backwards for our purpose".

What is available instead is the profile, and
`blossPowerlineContaminationAirborne2016` states it:

> When looking at one sounding (dB/dt vs. time) by itself, **it is nearly
> impossible to tell that the data are affected**; however, when looking at a
> profile of data (dB/dt vs. distance) the effect is quite prominent as a
> **symmetric double peak signature, extending up 300 m away from the powerline,
> and with soundings directly above the powerline largely undistorted.**

The pattern is characterised — symmetric, double-peaked, central trough, wider
at late time than early — and Asif add a fourth dimension: it **sweeps across
the gates** as the system passes, late gates lighting first on approach and
going out last on departure. That ordering is in the time–distance plane and is
free, because the data is already a 2D array of gate against distance.

A per-sounding detector run across this pattern does the wrong thing twice: it
flags the flanks and **keeps the centre**, leaving an island of apparently good
data directly over the conductor. Detecting the pattern rather than the sounding
is the contribution available here, and it is ours.

Physics, geometry and the window: **Appendix C**.

## What the detector cannot do

Four limits. The first two are deliberate, the last two bound where it applies.

**These are applicability conditions, not defects.** Every processing filter in
this library is selective — `cull_by_uncertainty` ships **no default** for
`max_uncertainty` because "what is tolerable depends on the inversion",
`qc_05_powerline` writes a verdict only where thresholds were supplied, and
`thresholds.py` refuses to classify a metric with no threshold rather than
guess one. A chain is composed per survey; nothing runs everywhere. So "not
the right tool on that survey" is an ordinary thing to say about a step here,
and the place to say it is the docstring, so whoever builds a chain knows
before they add it.

**1. Oscillatory coupling is out of scope.** A late-time sign reversal means
coupling, IP, *or* noise, and IP is geology — common geology, not exotic. The
reading is ambiguous and one reading is signal, so this detector targets smooth
elevation only. See **Appendix B**.

**2. IP gets culled anyway, by `cull_negatives`, and that is acknowledged.** We
cannot model IP yet, and data the forward model cannot represent cannot be
inverted correctly. The behavior is right; only the *account* of it needs
changing, in `filter_32`'s docstring. Same for 1D: a known flaw and still
industry standard. See **Appendix B**.

**3. A 1D reconstruction basis flags genuine 3D geology.** The criterion is a
test of one-dimensionality, not of cleanliness. Asif's auto-encoder has this by
construction — trained on a million **1D** forwards — and they state it:

> If there are 2-D or 3-D effects in the data that cannot be represented by any
> 1-D resistivity model, a low confidence value may be obtained for such data,
> which consequently might be identified as an anomaly by our expert system.

They also scope their work out of our harder case: "Our methodology is designed
in a hydro-geological context. If the infrastructure that induces coupling
response in the data is considered as exploration targets, a different strategy
may be required."

### 4. The double peak is not diagnostic

This is the finding that most constrains the design, and it was got wrong in an
earlier draft of this plan.

**A steeply dipping conductive sheet is a current loop in a vertical plane.** The
flux-cancellation argument that explains the central trough (Appendix C) is a
statement about geometry, not about infrastructure: directly above a vertical
conductor the induced current flows in a vertical plane, its field at the
receiver is horizontal, and the vertical component nulls. Off to either side the
coupling grows, peaks and falls away. **The same symmetric double peak with a
central trough, from geology, with no powerline present** — it is the textbook
signature of a steeply dipping conductor. Graphitic faults, shear zones, tilted
banded iron formation. Off-line conductors channel current into themselves as
well, so the body need not even be under the flight line.

So the profile criterion is not the safe one. Its false-positive class is **the
same class as the reconstruction criterion's** — conductive, compact, strongly
responding bodies, which is to say exploration targets. Both can cull the thing
the survey was flown to find, arriving by different routes.

**What separates a power line from a dipping conductor, and what does not:**

| | discriminates? |
|---|---|
| symmetric double peak with central null | **no** — a vertical sheet gives the same |
| strict symmetry | **partly** — a *tilted* body is asymmetric, so symmetry argues against dip, but a vertical graphitic fault is symmetric too |
| peak separation vs flight height | **weakly** — a shallow source gives a narrower pattern, a deep conductor a broader one |
| ruler-straight over kilometres, constant bearing across many flight lines | **somewhat** — geology is linear too, but it bends, splays and varies in strength along strike |
| the sweep-across-gates ordering | **unknown** — worth testing, see open questions |
| an infrastructure layer | **yes** |
| the powerline monitor | **yes, where it responds** |
| a height excursion over the crossing | **yes, where the pilot climbed** |

Everything reliable in that table is **non-geophysical**. That is the finding,
and three things follow:

1. **The monitor's role depends on context.** "Corroborates, never triggers" is
   right for hydrogeology, where missing a quiet crossing costs more than
   culling a conductor. In mineral exploration the costs invert, so
   corroboration becomes close to mandatory. That is the `corroborate` mode —
   a setting chosen per survey, like every other threshold here.
2. **Say in the docstring where this step suits and where it does not**, so a
   chain's author decides before adding it rather than after reading a model.
3. **The second pass earns its keep most where targets are expected.** A step
   that can discard one wants a human reading what it discarded.

None of which sinks the approach — it scopes it, which is the normal condition
of a filter here. Most surveys in hand are hydrogeological or sedimentary,
where a bedded clay layer will not imitate a vertical sheet. Where conductive
targets *are* expected, the honest options are to require corroboration or to
leave the step out of the chain. The detector cannot work out which setting it
is in, so the chain's author tells it — exactly as they already choose
`max_uncertainty` and every other threshold in this library.

### And two priors the detector must not carry

- **Resistive terrain is not safe.** Loop current is limited by contact
  resistance at the grounding points, not by the background — a 1000 Ω·m
  half-space with poles 30–40 m apart still gives single-digit ohms of earth
  return. Contact resistance is unknowable per pole, so there is no prior about
  which surveys or crossings will be affected. Look everywhere. **Appendix C.**
- **Neither moment is privileged.** The effect peaks in a *time window* near the
  LM/HM crossover rather than in a moment, so run on both. **Appendix C.**

## The systems: SkyTEM is the exception, not the template

All the major players are in scope — **SkyTEM, VTEM, HeliTEM and NRG** — and
almost every number in this plan comes from SkyTEM, which is the one that is
least like the others.

| | SkyTEM | VTEM, HeliTEM, NRG |
|---|---|---|
| moments | dual — LM and HM | **single**, all three |
| high-rate delivery | yes | **sometimes** — a 10 Hz product on VTEM, not guaranteed |
| per-gate error estimates | yes | **no, never** |
| in this plan | Kang's simulation and field data; Asif's airborne case | `bloss…2016`, `bloss…2017` (VTEM) |

**Single-moment is the normal case.** That simplifies two things and sharpens a
third:

- The LM/HM question (Appendix C) is a SkyTEM question only. Everywhere else
  the whole burden falls on one transient and there is no moment to choose.
- Network count, if the auto-encoder is ever built, is **one per system** rather
  than two — four networks for four vendors, not seven.
- But the three VTEM-like systems are also the three with **no error
  estimates**, so the modeled denominator below is not a fallback for an edge
  case. It is the path almost everything takes.

**And "raw" is not raw.** Every vendor applies a processing layer between the
instrument and the delivery, and what that layer does is not published. The
high-rate product is *minimally processed*, not unprocessed. Two consequences
run through everything below.

**The profile signature is the part that crosses systems.** The double peak with
a central trough was observed in VTEM data in the thesis and derived from 3D
simulation of a SkyTEM geometry by Kang, independently, and they agree. The
geometric criterion rests on the physics of the power-line loop rather than on
any one system's waveform, so it should port without retraining.

**The reconstruction criterion does not port**, for three reasons in increasing
order of difficulty:

1. **A single-moment system has no LM/HM contrast to exploit.** On VTEM the
   whole burden falls on one transient. It also halves the network count — one
   per system, not two.
2. **Retraining is required, but it is a forward-modeling job, not a labeling
   job.** Asif: the method "is largely insensitive to the system transfer
   functions … and can easily be extended for other TEM systems by retraining on
   forward responses for those systems." No labelled VTEM data is needed, which
   is fortunate, because we have none.
3. **Asif's method cannot run as published on VTEM at all.** They flag the
   possibility; for us it is the normal case:

   > Obtaining the raw data stacks and the corresponding data uncertainties is
   > not trivial and may not be available for some TEM systems on the normal
   > user/client level. In that case, the thresholds α and β_max may be
   > empirically adjusted to obtain adequate processing results.

   Their ε is a residual **in sigmas**, and the sigma comes from the scatter of
   raw transients within a short stacking window. **VTEM deliveries carry no
   error estimates**, so that denominator does not exist. SkyTEM carries both,
   but the uncertainties are the *vendor's*, computed over the *vendor's*
   stacking window — and Asif are explicit that the window must be "relatively
   short … to obtain good uncertainty estimates and to prevent the coupling
   signatures to be smeared." We do not choose it and are not told it.

**So the modeled denominator is the default path, not the fallback.**
`estimate_noise_model` followed by `assign_uncertainty` puts a modeled
uncertainty on every gate — the same denominator `compute_misfit` already
divides by. It is weaker, because a modeled sigma cannot see a stack that
happened to be noisy. It is also the only thing that works on VTEM, and it has
one real advantage: it is the *same* denominator across systems, so a threshold
calibrated once is not silently re-scaled by a vendor changing their stacking.

Where SkyTEM uncertainties exist they are worth using and worth comparing
against the modeled ones — the comparison is itself a QC check, and
`noise_floor_drift` in the compare group is close to the right shape for it.

### The black box is a systematic, and it is why stage 1 is a PCA basis

The vendor's pre-processing is an unknown change to the effective system
transfer function. Asif's tolerance — "largely insensitive to the system
transfer functions … **unless significant changes are made**" — is doing real
work in that sentence, and an undocumented filtering and stacking chain is
exactly the change it excludes.

This matters more for a network trained on forward responses than it looks,
because the mismatch is **systematic**: every delivered sounding differs from a
clean forward in the same direction, so every sounding reconstructs slightly
badly, and the confidence metric is depressed across the board rather than at
the couplings. That is survivable — a uniform offset only moves where the
threshold sits — but it means α and β_max must be recalibrated **per vendor and
system**, not per system, and it removes any value in copying numbers across
deliveries.

**A linear basis fitted on the delivered survey itself does not have this
problem.** It learns what delivered transients look like *in this survey*,
black box included, and the vendor's processing lands in the basis rather than
in the residual. That is a stronger argument for staging the PCA reconstructor
first than "it is cheap and needs no training" — on data we actually receive it
may be the more robust of the two.

**Its own failure mode is the mirror image**, and must be designed around: a
basis fitted on data that contains coupling will absorb the coupling too, if
there is enough of it to reach the leading components. Three mitigations, in
order of preference — keep the component count low enough that only the smooth
decay is captured; fit on line segments with no infrastructure within the
search window; or iterate, fitting, detecting, and refitting with the detections
excluded. Decide this when stage 1 is built; it is the one real design question
in it.

This is also why the profile criterion is staged second rather than third: it
needs no uncertainties at all, so the whole denominator problem passes it by.

## How this gets used: two passes

A QA/QC tool first and a processing step second, which changes what it must
produce.

**Pass one is unattended.** Data arrives, goes through the chain, inverts, and
produces a model and a misfit. Nobody adjudicates individual soundings. The
output is a defensible first answer and the evidence needed to question it.

**Pass two is guided by pass one.** A human reads the first result and adjusts:
add culled data back, leave it, or cull more by hand. `apply_manual_edits`
(filter_33) is already the checkpoint, and `version_diff` and `parameter_sweep`
already exist to show what changed.

### Pass one culls. It does not merely flag

**Decided.** The alternative — flag the suspect data, leave it in, let a human
decide later — is tempting because it cannot delete a target. It is wrong here,
for three reasons.

**The model is the QC instrument, so it has to be clean enough to read.** You
cannot look at a transient and pronounce it good; that judgement is only
available downstream, as whether the data inverted into something sensible. The
first-pass product *is* the model and its misfit, so anything that degrades the
model degrades the instrument. Leaving coupling in to be safe corrupts the very
thing pass one exists to produce.

**Coupling does not stay where you left it.** Unculled, it does not sit quietly
in a few soundings — the lateral and spatial constraints and the regularisation
spread it, which is the documented failure in
`blossPowerlineContaminationAirborne2016`: false conductors "get smeared out into
the model domain." A flag on three soundings does not contain an artifact that
ends up thirty soundings wide. Convergence suffers too, and Asif measured the
other side of it: inversion of automatically processed data "takes fewer
iterations to converge as compared to the manually processed data due to the
improved handling of the couplings."

**And the cull is reversible, so it is not the risky act.** Pass two reruns from
the original data regardless — culling is subtractive, so nothing is ever
un-culled in place — which means a pass-one cull costs nothing permanent. The
irreversible failure is not *removing* the target; it is **nobody noticing that
it was removed**. That is a reporting problem, and it is solvable.

So the obligation moves from "do not cull" to **"cull legibly."** Three
consequences, each cheap now and awkward to retrofit:

- **Write the confidence, not only the verdict.** Recording `C_avg` per sounding
  as a column alongside the mask gives pass two a continuous field to
  re-threshold and a figure showing *how close to the line* each sounding sat.
  Without it, "put some back" is a blind re-run; with it, it is a judgement
  about a visible margin. This is the one change that makes the two-pass
  workflow work, and it is one array.
- **The coupling cull must be distinguishable in the provenance.** The library
  already writes `why_` arrays with a step code, so this is nearly free — but
  pass two has to find *coupling* culls specifically, among everything else that
  culled a gate, to reverse them selectively.
- **What the filter took must appear in the report.** `culled_fraction` (qc_12)
  exists; it needs to break out this step so a reader sees that the coupling
  filter removed 4% of the survey, and where. A cull that is reported is a cull
  somebody can argue with.

**Default to Asif's calibrated thresholds**, not deliberately light ones. An
earlier draft argued for a light hand because over-culling is invisible — but
that is the premise the three points above remove. With the confidence column
and the culled-fraction report, over-culling is visible, while under-culling is
the one that quietly smears. The guard against discarding real targets is
the `corroborate` mode where geology can imitate the pattern, not a globally
timid threshold.

## Shape of the step

### The step owns its own table, and the switch decides whether it is applied

**Decided.** The detector writes **one table of its own**, the same shape as the
mask, every run. All the work lands there. The parameter only decides whether
that table is then applied to the in-use mask:

```yaml
- cull_by_coupling:
    action: flag          # flag | cull
```

| `action` | writes the coupling table | writes `inuse_{channel}` |
|---|---|---|
| `flag` | yes | no |
| `cull` | yes | yes |

**Why a separate table rather than two separate steps.** An earlier draft here
proposed two registrations — a QC check that flags and a filter that culls —
over a shared detection core. That was more machinery for the same result. With
one table there is **one code path**: the detector always does the identical
thing, and `cull` adds a final, trivial, auditable operation on top. Nothing can
drift between a flagging run and a culling run, because there is nothing to
drift.

It also gives for free the property the two-step version had to arrange
deliberately: **the table survives both modes**, so a culling run still carries
the full record of what the detector thought, which is what pass two needs in
order to reverse anything.

**On the library's governing rule.** `core/qc/__init__.py` says "QC writes
attributes. Culling writes the mask." This respects it: the table is an
attribute, and only `action: cull` touches the mask. A filter that declines to
filter is a little unusual, but it is honest and it is one step with one
description, which is better than two steps that must be kept in step.

#### What the table holds

Same shape as `inuse_{channel}`, one per channel, under a **prefixed** key so it
stays outside the anchored channel-discovery patterns — the lesson `why_`
already encodes, and the reason it is a prefix and not a suffix.

**Int8 rather than boolean**, mirroring the provenance encoding:

```
 0  clean
+1  detected directly
-1  swept up by the culling policy (spread_earlier, disable_tail)
```

Boolean would do, but the sign costs nothing and carries something a reviewer
wants: whether a gate was *found* or merely *adjacent to* something found.
`cull()` already draws exactly this distinction, giving direct culls `+code` and
collateral `-code`, so the table and the provenance speak the same language.

**It holds the post-policy answer** — what *would* be culled, not the raw
predicate. The point of flagging is to show what the cull would do, so the
policy has to have already run.

**Applying it is one call**, because the mask API was built for this:

```python
cull(dataset, channel, bad=table != 0, step_code=NN, direct=table > 0)
```

`bad` is the policy's output, `direct` the predicate's own answer. Provenance,
the subtractive guarantee, and the per-channel counts all come free.

#### Boolean, not a stored residual — and the reason is a measurement

A per-gate ε would let pass two re-threshold without re-running the detector,
and the storage is trivial. It is still not worth it: a PCA reconstruction over
90,000 soundings is **seconds** (Asif measured 0.3 ms per stack for a heavier
network), so re-running with a different threshold is cheap enough that caching
the continuous field solves a problem nobody has. The per-sounding confidence
column stays, because it drives the rollups in `aggregate.py` and is what
somebody scans to find *which* crossings to look at — it is the index; the table
is the payload.

#### Why not a per-sounding verdict column

Every QC check so far judges **auxiliary data** — altimeter, tilt, speed,
current — single channels that describe a whole sounding, so a verdict column on
`flightlines` fits exactly. **Coupling is not like that.** It affects a band of
gates, the band moves with distance from the line, and the whole time-window
finding (Appendix C) says *which* gates is the information. One value per
sounding throws that away, which is why the table is per gate.

#### The per-gate view is the easy one to draw, not the hard one

Worth checking before assuming this is expensive. `ChannelPlot` already renders
a **four-state per-gate overlay**, GPU-shaded:

```
0 = auto-enabled       1 = manually-enabled
2 = auto-disabled      3 = manually-disabled
```

State is computed per sounding **per gate**, from the committed mask overlaid
with a pending in-memory edit. So the plot already distinguishes *what the
pipeline did* from *what a human proposed*, at gate resolution, today.

The coupling table is a third category of the same kind — proposed by the
algorithm rather than by a person — so it is **one more state in a shader that
already branches on four**, fed by an array shaped exactly like one it already
reads. Two, if the direct/collateral sign is drawn differently, which is worth
doing: "found here" and "removed because the tail rule said so" are different
things to agree or disagree with.

By contrast, **nothing in the frontend reads `qs_` columns at all** — a search
returns nothing, for any QC check. So the per-sounding verdict route, which
looked like the cheap one, has no rendering whatsoever, while the per-gate route
is mostly there. That is the opposite of the intuition and it decides the design.

#### Which closes the loop with the manual-edit machinery

The frontend's edit format is `layer_data["InUse_{channel}"]` with **NaN meaning
no override**, addressed by `flightlines.apply_idx`, marked by
`model_info.diff_dummy`. **The coupling table converts to that object** in a few
lines — 0 where it is non-zero, NaN elsewhere, `apply_idx` from the rows that
have any entry.

So a flagging run can emit a diff that the existing editor loads and renders,
a person brushes to adjust, and `apply_manual_edits` (filter_33) applies —
through machinery that is already built and tested. The detector's proposal and
a human's correction become the same kind of thing, which is exactly the
two-pass workflow, and nothing new has to be invented to carry it.

The one gap: the frontend builds that overlay from its own in-memory reducer,
not from a diff loaded off storage. Giving it a path to load a pipeline-emitted
diff is the real front-end work here — smaller than a new visualization, and it
makes every future proposing filter reviewable the same way.

### `qc_05_powerline` has already solved the monitor baseline

Worth knowing before writing any of the corroboration code. It finds **local
peaks** in the monitor — "each sounding is compared to its neighbours — a
rolling window along the line — rather than to the line as a whole" — because
"the ambient level moves within a line", and a line flown from farmland into a
town would otherwise read as one long spike.

That is precisely the relative-not-absolute reading the PLNI section argues
for, already implemented. **The corroboration check should consume qc_05's
excess column rather than re-deriving a baseline**, which also means the two
agree by construction instead of by coincidence.

### The culling step

One filter, several criteria, following `cull_by_external_evidence` — which
already takes a `criterion` and has the `proximity`/`misfit`/`both` precedent.

```yaml
- cull_by_coupling:
    action: flag                  # flag | cull
    criterion: reconstruction     # reconstruction | profile | trend | both
    max_residual: 2.5             # sigmas; Asif's alpha
    confidence: 0.99              # Asif's beta_max
    window: 7                     # soundings smoothed over; Asif's W
    max_uncertainty: 0.2          # mask noise-dominated gates; Asif's delta_max
    confidence_column: qs_coupling  # write C_avg, so pass two can re-threshold
    corroborate: none             # none | plni | infrastructure | any | all
    infrastructure: null          # vector geometry, as filter_30 takes it
    spread_earlier: 1
    disable_tail: true
```

#### Where it sits in the chain

`docs/reference/llms.txt` keeps an **Ordering** section, opening "a wrong order
here does not fail. It produces plausible nonsense." Two placements are forced
rather than preferred, and both belong in that list:

- **after `assign_uncertainty`** — ε divides by Δd_obs, and before that step
  there is nothing to divide by. The reconstruction criterion cannot run
  earlier; it would have to invent a denominator.
- **before `average_along_line`** — exactly the reason `cull_negatives` is
  ("cull before averaging, or a bad gate is smeared into its neighbours"), and
  sharpened here by Asif's own warning that too long a window "would smear the
  effect of couplings in the data." Averaging first does not merely dilute the
  signature; it spreads a coupled sounding into its clean neighbours and makes
  the profile shape the detector is looking for *less* distinct.

So: `estimate_noise_model` → `assign_uncertainty` → `cull_by_coupling` →
`average_along_line`, with `apply_manual_edits` last as always.

#### It must work without a shapefile. Decided

**The detector is built to need no infrastructure layer.** A layer may be
supplied and will raise confidence where it exists, but nothing in the design
may require one, and no capability may be available only with one.

The reason is coverage. Layers are patchy, stale, and absent far more often
than not — Kang had to find lines in satellite imagery that were in no
database, and still warned that others were hidden by trees. A tool that works
only where the map is good works in the minority of cases, and would be useless
on exactly the surveys where nobody has mapped anything.

This resolves what was an open question. **`cull_by_external_evidence` is
already the shapefile path**: filter_30 culls by proximity to mapped features,
widening the exclusion with gate time in `log t`, per moment. It exists and it
works. So this step is not a competitor to it and must not be folded into it —
it is the path for **when there is no layer**, which is the common case and the
one nothing currently covers.

#### The real difference: the map as instruction, or as evidence

"Mapped versus unmapped" undersells it. `cull_by_external_evidence` is
**purely geometric** — it culls everything within a distance of a mapped
feature, having looked at no data at all. That is a blunt instrument in both
directions, and systematically so:

- **it over-culls the quiet crossings.** A pole with degraded grounding may
  produce almost no coupling, and a fixed buffer discards good data there
  anyway. Kang's own field result is the measure of it: 174 m predicted,
  ~104 m realised, because the grounding decides and the grounding varies pole
  to pole.
- **it under-culls the loud ones**, where a large loop and good grounding push
  the effect past the configured distance.

So the distinction between the two steps is not which features they cover. It
is what the map *is*:

| | the map is |
|---|---|
| `cull_by_external_evidence` | an **instruction** — cull within `D`, regardless of the data |
| `cull_by_coupling` | **evidence** — the data decides, the map raises confidence |

That is the more useful framing, and it means the detector is worth preferring
**even where a good layer exists**, with `corroborate: infrastructure` set,
because it culls what actually coupled rather than everything inside a buffer.
filter_30 keeps its place as the cheap, deterministic, auditable option, and
for cases where a zone must be removed whatever the data says.

#### And its `misfit` criterion is backwards for our purpose

Worth stating because it looks like it should already solve this. The `misfit`
criterion culls gates a previous inversion fitted poorly, beyond
`max_misfit` standard deviations. That finds the wrong half of the problem:

- **wild coupling** — oscillatory, implausible — no 1D model fits it, misfit is
  high, and the criterion catches it. But the shape filters already catch that.
- **smooth inductive coupling** — the actual gap — is a plausible-looking
  elevation that a 1D inversion fits *comfortably*, as a conductor that is not
  there. **Low misfit, bad model, and the criterion passes it.**

Asif describe the same thing from the other side: the coupling "appears as an
enhanced signal level, which could easily be interpreted as a geological
change." Interpreted as a geological change means fitted by a model, which
means low misfit.

**So a low-misfit coupled sounding is the dangerous one**, and any detector
built on misfit is selecting against exactly the case that matters. This is an
argument for staying in the **data domain, before inversion**, and it applies
to `data_misfit` (qc_11) in the two-pass workflow as well: a quiet misfit map
on pass one is not evidence that the data was clean.

Two consequences recorded elsewhere follow from this decision rather than
standing on their own: the crossing angle is recovered from the **trend**
rather than read from a layer, and candidate discovery is by **collinearity**
rather than by proximity to mapped features.

**The reconstructor is swappable behind one interface.** `reconstruct(transient)
-> transient` is all the criterion needs; the confidence metric, the smoothing
and the fuzzy decision sit on top of it unchanged. So:

- a **linear basis** (PCA/SVD over the gate dimension, after
  `xieImprovedPrincipalComponent2014` and the SVD lineage Andersen cites) works
  today — no training, no torch, no model artifact;
- the **auto-encoder** drops in later without the pipeline step changing.

Asif's decision machinery is worth copying closely, because it is the part that
turns a residual into a decision:

- mask gates beyond the first where uncertainty exceeds `Δ_max` (20%) — that is
  noise, not coupling, and not this step's business
- `ε(i) = |d_recon(i) − d_obs(i)| / Δd_obs(i)` — residual **in sigmas**, the same
  unit `compute_misfit` already uses
- `C(α) = (1/N) Σ [ε(i) < α]` — the fraction of gates that reconstruct
- smooth `C` along-line over `2W` before deciding, so no single sounding decides
- a fuzzy band rather than one threshold: clearly high `C` keeps everything,
  clearly low culls everything, and between them only gates with `ε ≤ 1` survive

### The heavy hand and the light hand are two of these parameters

Asif expose four, and describe them in exactly the terms the second pass needs
(Section V):

> The automated processing involves several parameters, i.e., α, β_max, W, and
> Δ_max. **The thresholds α and β_max give an intuition about the leniency or
> strictness in the automated data processing.**

So the control surface already exists in the published method, and it is two
numbers rather than a vague dial:

| parameter | theirs | what moving it does | heavier hand |
|---|---|---|---|
| `max_residual` (α) | 2.5 | how far a gate may miss, in sigmas, and still count as reconstructing | **lower** |
| `confidence` (β_max) | 0.99 | what fraction of gates must reconstruct for the sounding to be kept | **higher** |
| `window` (W) | 7 stacks | smooths short-term fluctuation before deciding | neither — see below |
| `max_uncertainty` (Δ_max) | 20% | where a gate stops being data and becomes noise | **lower** |

Three qualifications, all theirs:

- **`W` is not a strictness dial and should not be used as one.** "A larger
  value of W would smear the effect of couplings in the data" — it is set by the
  stacking rate and the along-line sounding spacing, and raising it to get a
  quieter answer destroys the signature being looked for.
- **`Δ_max` is self-correcting in one direction.** A higher value admits gates
  affected by background noise, but "the values of ε for such data points will
  still be low due to their higher data uncertainty and they will be classified
  as uncoupled." Loosening it is safer than it looks; tightening is the sharper
  move.
- **They are not twitchy.** "These parameters are not very sensitive and can be
  adjusted in different ways to obtain similar data processing results", and a
  grid search across several survey areas found one set of thresholds matched
  manual processing best in all of them. The caveat is ours, though: "This
  scenario may not be true if the signal levels are drastically low, e.g., in a
  very resistive case."

That last point and the raw-stack problem meet in the same place. Asif's advice
where stacks and uncertainties are unavailable is that "α and β_max may be
empirically adjusted" — the published defaults are tied to *their* uncertainty
estimates, so on a delivery where we model the sigmas instead of measuring them,
the two thresholds must be recalibrated rather than copied. `parameter_sweep` is
the tool, and the recalibration is per system, not per survey.

**The step's defaults should still be Asif's**, because they are the only values
with evidence behind them — with a docstring note that they were calibrated
against measured stack scatter on tTEM and SkyTEM, and that a different
uncertainty source moves them.

### Corroboration

In hydrogeology the detector decides alone: no two crossings are alike, a pole
with poor grounding may produce almost no monitor response, and a detector gated
on a witness would miss the quiet crossings entirely. Where a witness exists and
agrees, confidence rises; where it is absent — 3133 soundings in the one real
delivery have a flat monitor — the data still decides.

**Where geology can imitate the pattern, that inverts**, and the witness becomes
a precondition rather than a bonus. So corroboration is a mode, not a boolean:

```yaml
corroborate: none        # none | plni | infrastructure | any | all
infrastructure: null     # optional vector geometry, as filter_30 takes it
```

**`none` is the default and the design target.** The modes requiring a layer
are available where one exists; nothing depends on them — see "It must work
without a shapefile".

| mode | cull when the pattern is found and… | |
|---|---|---|
| `none` | always | hydrogeology default |
| `plni` | the monitor agrees | |
| `infrastructure` | a mapped feature is within range | |
| `any` | either agrees | |
| `all` | both agree | strictest; protects targets hardest |

This splits the ambiguity in "PLNI and shapefile": **`any` and `all` are
different tools.** `all` is the conservative one — two independent witnesses
must line up before anything is removed — and is the right default where the
cost of culling a target is catastrophic. `any` is the middle setting, for a
survey where the infrastructure layer is known to be patchy and the monitor
known to be insensitive, so either alone is worth acting on.

**The infrastructure layer needs almost nothing new.** `filter_30` already takes
vector geometry as a *type* (a `GeoDataFrame`, with a path accepted for
standalone runs), and already carries the thing a naive proximity test would get
wrong: the exclusion distance **widens with gate time**, interpolated in `log t`
between a first-gate and last-gate value, **per moment**, because late-time data
senses a larger volume and is affected from further away. The coupling step
should reuse that distance logic rather than reimplement it — the difference is
only that `filter_30` culls on proximity while this queries it.

It is also the one witness that is wholly independent of the geophysics, which
is what makes it the most valuable. Its weakness is coverage: Kang had to find
lines in satellite imagery that were in no database, and still noted "there
could be other power lines that were not observed due to poor satellite imagery
or other objects (e.g., trees) obscuring the power lines." **A missing line is
normal; absence from the layer is not evidence of absence.**

### How the PLNI channel was made changes what it is worth

It changes it a lot, and the plan should not treat "the monitor" as one thing.
Two vendor practices are in circulation:

| | separate coil tuned to 50/60 Hz | bandpass of a late gate on the main receiver |
|---|---|---|
| independence | a genuinely separate sensor | derived from the same data stream being judged |
| sensitivity | optimized for the frequency | works *against* the system's own 50/60 Hz rejection |
| spatial response | its own position and orientation, so its peak need not coincide with the TEM receiver's | the receiver's |
| failure at the crossing | degrades independently | degrades with whatever disturbed the stack |

Three consequences:

- **Corroboration is only worth something if it is independent.** A PLNI
  bandpassed out of the same late gates the detector is examining is weaker
  evidence than it appears, because a disturbance that moves one can move the
  other. A tuned coil does not have this problem.
- **A bandpass-derived PLNI is insensitive by construction.** TEM suppresses
  50/60 Hz at acquisition deliberately — synchronous detection with alternating
  transmitter polarity — so recovering a powerline field from the main receiver
  means working against a rejection the system was designed around, with a
  rejection ratio that varies with how well the base frequency is locked to the
  mains. It will miss quiet lines that a tuned coil would see.
- **So the sensible mode depends on which PLNI the survey has.** With a tuned
  coil, `corroborate: plni` is reasonable. With a bandpass-derived channel,
  requiring it can block legitimate culls, and `infrastructure` is the better
  witness. **Record which one a survey carries**; it is a property of the
  delivery, not of the step.

**And in both cases the PLNI measures the wrong thing, slightly.** It senses
the line's own 50/60 Hz emission, which says the line is present and energized
— not that it is coupling to our transmitter. A de-energized line still couples
inductively; a heavily loaded line with corroded grounding may emit strongly and
couple weakly. The two are correlated, not the same, which is the deepest reason
the monitor corroborates rather than triggers.

Practically, that means the PLNI must be read **relatively** — elevated against
this line's own baseline — never against an absolute threshold, because its
calibration is vendor-specific and undocumented.

### Height excursion

Worth wiring in as a third witness: `height_above_ground` (qc_01) already
computes it, pilots climb over powerlines, and the climb is coincident with the
crossing by construction. It is the weakest of the three — not every crossing
gets a climb, and terrain produces climbs of its own — so it belongs as an
advisory input that raises confidence rather than as a mode of its own. If it
later earns one, the enum above extends without changing shape.

## Designing the profile criterion

The reconstruction criterion is specified completely, because Asif specified
it. The profile criterion — the part that addresses the actual gap, with
nothing to copy — was until now a description of a pattern rather than a
method. This section is the method, scoped deliberately small.

**Build the shed first.** Z component only, no three-component data, no
template matching, no simulation required to start. The extensions have their
own section at the end and stay there until the shed stands up.

### What sampling allows, which is less than the physics suggests

The design has to start here, because it rules out the obvious approach.

Take Kang's distribution line: null at 0, peaks at **±30 m**, falling to 3% by
**±174 m**. At a 30 m sounding interval that is ~12 samples across the whole
affected zone but only **about three across the double-peak core** — and if the
sampling happens to be offset by half an interval, neither the null nor either
peak is ever sampled, only four points partway up and down the flanks.

So:

> **Symmetry survives sampling. Structure does not.**

Any detector that matches a *double-peak shape* will fail at ordinary sounding
spacing on ordinary lines. The detector therefore looks for **a locally
confined, symmetric departure from the along-line background** — a claim that
holds whether the core is resolved, half-resolved or not resolved at all, and
whether the line is distribution or transmission.

This also disposes of the worry that a sounding will rarely sit exactly on the
null. It never needs to.

### 1. Locating the crossing: the axis of symmetry, not the null

**Do not search for the trough.** Its position is quantised by the sounding
interval at best, and it may fall between samples entirely.

Instead, estimate the **axis of symmetry**: cross-correlate the profile segment
against its own reverse, and the lag that maximises the correlation is twice
the offset of the symmetry axis from the segment centre. This uses the
**flanks**, which are well sampled, to locate a centre that may never be
sampled, and it gives sub-sample precision — on the order of ±5–10 m for a
clear anomaly, against a collinearity test that operates over kilometres.

Three properties make this the right primitive:

- **It is altitude-invariant.** The symmetry axis sits over the line at any
  flight height. Height changes the *width* of the pattern, not its centre —
  so varying altitude between lines, which is certain, does not move the thing
  being measured. The widths are not compared across lines; only the positions.
- **It is angle-invariant.** A straight flight path crossing a straight power
  line is symmetric about the crossing at *any* angle. Only the apparent width
  stretches, by `1 / sin θ`.
- **It yields its own uncertainty**, from the sharpness of the correlation
  peak, which the collinearity fit below can weight by.

Each candidate therefore carries: a **position** (map coordinates, sub-sample),
a **measured width**, an **amplitude**, and a **position uncertainty**.

### 2. Width is measured, never predicted

Three independent reasons now converge on this, and it is worth stating once:

- **contact resistance** is unknowable and governs the realised extent
  (Appendix C);
- **line class** is often unknown, and loop area spans an order of magnitude;
- **crossing angle** is unknown, and stretches apparent width by `1 / sin θ`.

So the derived extent from Appendix C is used only to **bound the search
window**, never as a threshold or a template. The measured width is an output.

### 3. Collinearity, across every line regardless of bearing

Candidates from all lines go into one map-coordinate set, and the test is for
**collinear subsets with small residual scatter**. This is a Hough transform or
RANSAC — each candidate votes for the straight features through it, and peaks
in the accumulator are trends.

**No distinction between flight lines and tie lines.** The test works on
`(x, y)` points and has no reason to know a line's role or bearing. Tie lines
are not assumed perpendicular, are not required, and a survey without them
loses nothing structural — lines at differing bearings simply give better
geometry for the fit.

**The test is on residual scatter, not on collinearity existing.** A
transmission line is straight to within metres over 10 km; geology bends,
splays and varies along strike. That difference is a number with a threshold,
which is what makes this a discriminator rather than an observation. Its
honest limit: it catches transmission lines well and **road-following
distribution poorly**, because roads curve.

**Missing lines are expected and harmless.** Hough and RANSAC both tolerate
gaps by construction, so a line flown high that failed to trigger simply does
not vote. And because `height_above_ground` (qc_01) is already computed, a
non-detection on a high line is *explicable* rather than evidence against the
trend.

#### Which is what lets the per-line detector be permissive

**The most useful consequence in this section.** If collinearity does the
discriminating, the per-line pass does not have to be precise. Run it with a
low threshold, over-produce candidates, and let the trend test sort them.

**Sort, not reject** — see §5. A candidate that fails the collinearity test is
not thereby noise: an isolated cultural source is non-collinear by nature, and
rejecting everything that does not line up would discard that whole class.

That matters because designing a precise per-line detector is the hard part,
and this removes the requirement. It also inverts the usual tuning problem: the
per-line stage is tuned for **recall**, the trend stage for **precision**.

#### What it gives back

- a **bearing** for each trend, from which the crossing angle `θ` follows per
  line, and from which the measured width converts to a perpendicular extent;
- **candidate discovery with no infrastructure layer**, which was an open
  question with no proposal behind it;
- **orientation independence**, which is the only partial answer to the
  parallel-line problem below.

#### Its one structural cost

The trend test is **survey-level**, not line-level, unlike every other filter
in the library. It needs every line present in one dataset. If a survey is ever
processed line-by-line, this criterion silently cannot work — and "silently" is
the dangerous part, so it must detect that it has only one line and say so.

### 4. The failure mode that stays open: lines parallel to the flight direction

The per-line detector compares a sounding to its along-line neighbours. If the
power line runs *along* the flight line, there is no uncoupled background to be
anomalous against: the whole line is elevated, nothing stands out, and the
detector reports clean with no sign that it failed.

Not an edge case — lines follow roads and valleys, and survey grids are often
laid out against the same terrain. It is `qc_05`'s "farmland into town" problem
with no return to baseline.

Partial answers, none sufficient alone:

- **Lines of a different bearing** in the same survey still cross it, and the
  trend test consumes them equally (§3). This is the main mitigation, and it is
  why not distinguishing tie lines matters.
- **An infrastructure layer** gives the bearing directly.
- **At minimum, refuse to claim a clean verdict.** Where a mapped feature or a
  recovered trend runs within some angle of the flight direction, the detector
  should report that it cannot judge rather than report nothing found. A hole
  that announces itself is survivable; a confident pass is not.

### 5. It is not always a power line

The design above is power-line-shaped, and most cultural sources are not power
lines. Sorting them by *what the detector can do* rather than by what they are
gives four regimes:

| regime | examples | profile | collinearity | reconstruction |
|---|---|---|---|---|
| **resolved, straight** | transmission, **railway** | yes | yes | yes |
| **resolved, curving** | **guardrail**, **debris fencing**, distribution following a road, farm fencing | yes | no | yes |
| **areal / unresolved** | **vineyards and trellised orchards**, towns, landfills, industrial sites | **no background** | no | fires everywhere |
| **point** | wreckage, well casings, tanks, silos, pivots | **too few samples** | **rejects it** | **the right tool** |

**Railways are the best case in the plan** — straighter than power lines,
continuous welded steel, bonded for signalling, grounded at intervals. If
collinearity works anywhere it works there.

#### The trend test needs a baseline length, not the whole survey

A guardrail is straight over 200 m even where it wanders over 2 km. So the
collinearity fit should run over a **configurable baseline** — a few adjacent
lines at a time — with the scatter tolerance scaled to it:

- long baseline, tight scatter → transmission lines and railways;
- short baseline, loose scatter → guardrails, farm fencing, distribution
  following a road.

Same algorithm, one parameter. This replaces what was previously recorded as a
flat limitation ("catches transmission well, road-following distribution
poorly") with a knob.

#### Areal sources: detect the edges, then fill

Vineyard trellis rows sit at 2–3 m and the soundings at ~30 m, so the rows are
far past Nyquist and are never resolved. What is seen is a **plateau over the
block with a step at its boundary** — and the same is true of a town, a
landfill or an industrial site.

That suggests the areal treatment is not pattern detection at all but
**edge-detect-then-fill**: find the entry and exit steps along each line, and
mark everything between them. It is a simpler algorithm than the symmetric
search, and it partially rescues the parallel-line failure in §4, because a
*region* has edges even when a line inside it has no usable background.

#### Point sources, and the limit of the whole approach

A compact conductor at 40 m flight height has a footprint of perhaps 50–100 m,
which at 30 m spacing is **two to four soundings** — far too few for the
symmetry estimator. The profile machinery does not apply to point sources at
all. The reconstruction criterion does, and does it well, because a confined
conductor produces a transient no 1D earth response resembles. This is the
clearest justification for carrying both criteria.

**But it is also where the plan's central problem is sharpest.** An isolated
cultural conductor and an isolated mineral target are *the same measurement* —
compact, conductive, failing to reconstruct against a 1D basis describes metal
wreckage and a massive sulphide body equally well. For power lines there was at
least collinearity and an infrastructure layer. For a pile of scrap in a field
there is neither, and **there is probably no geophysical answer**. Saying so is
better than implying one.

Three weak discriminators exist and are worth testing, none decisive:

- **Time constant.** For a compact body τ ≈ μ₀σa²/π². Metal at σ ≈ 10⁷ S/m
  gives τ of order *seconds* — effectively non-decaying across the whole gate
  window — while a sulphide body at σ ≈ 10 S/m and 50 m scale gives a few
  milliseconds, squarely in range. **An anomaly that does not decay at all over
  the measurement window is metal, not geology.** The strongest of the three,
  and arithmetic that should be checked rather than trusted.
- **Shallowness.** Wreckage sits at surface and hits the early gates hardest;
  targets are usually deeper.
- **Spatial coincidence** on adjacent lines — not collinearity, but the same
  body at the same map position. Only available where line spacing is fine
  relative to the footprint, which at 100–500 m spacing it usually is not.

#### Which forces one correction to §3

The trend stage was described as filtering: permissive per line, precise across
lines. **RANSAC discards non-collinear candidates by construction**, so as
written it would systematically throw away every point source — the entire
fourth regime, invisible by design.

So the trend stage must **classify rather than filter**:

| | verdict |
|---|---|
| collinear over its baseline | confident; infrastructure; safe to `cull` |
| isolated | uncertain; `flag` for review regardless of the step's mode |

That maps onto `action` and onto the int8 table already, and it matches the
real asymmetry in what we know: a 10 km straight trend is a power line, and one
compact anomaly is an open question.

#### A case the parameters do not cover

**Debris-fall fencing on a steep slope** is close to the worst case found so
far, and is recorded as unsolved rather than parameterised. It is rockfall
*mesh* — a two-dimensional conductive sheet, not a wire — draped on a hillside,
so it presents as an **inclined conductive sheet**, which is precisely the
dipping-conductor mimic the plan already cannot discriminate. It follows
contours, so it curves. And it lives where terrain is steepest, which is where
flight height varies most and the 1/r⁶ amplitude swing is largest. The hardest
cultural sources sit in the hardest terrain.

### Later, once the shed stands

Neither of these is part of the first build.

**The horizontal component.** Deliveries carry one — the real SkyTEM delivery
ships `HM_X_G01` — and the library already keeps it as an *unclaimed* array
rather than a channel. The trap to know about: **do not declare it in the GEX**
to make it available. `DualMomentTEMXYZSystem` hardcodes `rx_orientation='z'`
and would read a declared X channel as Z, which is a confident wrong model
rather than a refusal, and there is a test asserting exactly that. The right
route is to read it through `unclaimed_arrays`, which keeps it away from the
inversion. It must also stay optional and degrade loudly, because
single-component receivers exist.

What it might buy is **a sharper position estimate**. Reasoning from the
geometry rather than from a source: the response goes as (Tx→loop) ×
(loop→Rx), both antisymmetric about the crossing, whose product is symmetric
and positive — which is why the observed Z anomaly is a symmetric positive
double peak rather than a sign reversal. If that is right then the central null
is partly an *excitation* null, **no receiver orientation fills it**, and an
earlier draft of this plan was wrong to say X would. X should instead be
antisymmetric with a sharp **zero crossing** at the line, and zero crossings
localise better than broad troughs. It also predicts that concentric systems
(VTEM) and offset systems (SkyTEM) differ, because their two nulls coincide in
one case and separate in the other.

**That paragraph is derived, not read, and must be checked before anything
rests on it** — the same SimPEG run already queued for the dipping-plate test
can settle it.

**The sweep-across-gates ordering.** Still the only candidate discriminator
that is geophysical rather than circumstantial, and still untested; see open
questions.

## Cost, for a 2700 line-km survey at 30 m

**Inference is free. The million forwards are training, paid once.**

| | |
|---|---|
| soundings | 2700 km ÷ 30 m = **90,000** |
| passes | × 2 moments = **180,000** |
| per pass | ~97k MAC — 24-point input, three 32-filter conv layers |
| total | **≈35 GFLOP**, ~17 MB — seconds on CPU |

Asif measured it rather than estimating: **0.3 ms per raw data stack** on one
Xeon Gold 6132 core, fast enough that they propose running it in the instrument
in real time. At that rate the survey above is **~54 s on a single CPU core**,
which agrees with the arithmetic and settles the question — inference cost is
not a consideration in this design.

Training: Asif's 1M TEM-resolvable model dataset is **publicly available**, so
the expensive half — forward-modeling von Kármán structures and then *inverting*
them to get resolvable models — may not need regenerating. Their network took
~13,000 epochs at ~6 s/epoch ≈ **22 GPU-hours, once**.

The recurring cost is **one network per system per moment**: SkyTEM LM and HM
are two; HeliTEM, VTEM and Xcite would each need their own. A release-time cost,
not a per-survey one, and where a GPU forward model would pay for itself.

## How far is this from Asif's result?

Shorter than the length of this document suggests, because the document is
mostly about everything *beyond* Asif — the profile criterion, collinearity,
the geometry regimes — none of which that paper attempts.

**Asif's method itself maps almost one-to-one onto parts already built:**

| what the method needs | what exists |
|---|---|
| per-gate uncertainty, as ε's denominator | `estimate_noise_model` → `assign_uncertainty` |
| mask gates past the first above Δ_max | `cull_by_uncertainty(max_uncertainty, …)` — the same parameter, same meaning |
| culling policy — widen earlier, disable tail | `spread_earlier`, `disable_tail`, on every culling filter |
| subtractive culling with per-datum provenance | `mask.cull(…, step_code, direct=…)` |
| what was culled, reported | `culled_fraction` (qc_12) |
| a before/after figure | `figures.CULLING` |

**What is genuinely new is small:** a reconstruction basis, and the arithmetic
that turns a residual into a decision. The second is fully specified in the
paper — ε, `C(α)`, smoothing over `2W`, the fuzzy band — and is perhaps fifty
lines. The first is PCA/SVD over the gate dimension, which needs no training,
no torch and no model artifact; the family of 1D responses is smooth and
probably low-dimensional in `log`–`log`, so a handful of components may capture
it, and Andersen's own lineage used SVD.

**Where the time actually goes is integration, not algorithm.** The coupling
table is a new array family, and this library is strict about those: it needs
a prefixed naming pattern that cannot collide with anchored channel discovery,
it must not surface in `unclaimed_arrays`, `inventory.take()` and
`unclaimed_arrays` must agree about it (there is a test asserting exactly
that), it must survive export, and it needs a declared figure shape and a
regenerated reference or the suite fails. Several days of careful work, mostly
spent meeting the library's standards rather than on the detector.

**The distance that is not code is the real one.** Three things separate a
working implementation from Asif's *result*:

1. **Their 96% is not reproducible here.** It was agreement with their expert,
   on their surveys, with measured stack scatter. Ours would be a different
   number against a different judgement.
2. **Their thresholds do not transfer.** α = 2.5 and β_max = 0.99 are tied to
   measured uncertainties; against a modeled sigma they must be recalibrated,
   per vendor and system. That needs a real delivery and a line somebody has
   marked up by hand.
3. **Their method does not address the gap this plan is about.** Reaching Asif
   gets the reconstruction criterion, which by their own account cannot
   distinguish smooth inductive coupling from a geological change. Stage 2 is
   the part with no precedent.

**So: Asif's capability is stage 1, and stage 1 is close.** The implementation
is days; the calibration is gated on data rather than effort; and the thing
that takes real time is the half of the problem that paper leaves open.

## Staging

0. **The table and the switch.** The coupling table, written every run, and
   `action: flag | cull` deciding whether it reaches the mask. Worth doing
   first and almost free once the detector exists, because it makes every later
   stage reviewable rather than merely runnable. Converting the table to the
   existing diff format makes it editable in the editor that already exists; the
   front-end work is a path to load a pipeline-emitted diff, plus one or two
   more states in `ChannelPlot`'s overlay.
1. **The framework, with a linear reconstructor.** Residual in sigmas, the
   confidence metric and column, along-line smoothing, the fuzzy decision, mask
   and provenance. No new dependency. Testable against synthetic couplings now.
   Fitted on the delivered survey, it also absorbs the vendor's pre-processing
   instead of mistaking it for an anomaly — see "The black box is a systematic".
2. **The profile criterion, Z only** — a locally confined symmetric departure
   found per line, located by its axis of symmetry, then tested for
   collinearity across every line in the survey. Works on both systems and on
   delivered data, and needs no uncertainties, which is why it comes before the
   auto-encoder. Designed in "Designing the profile criterion"; still the
   riskiest stage, because it is the only part with nothing to copy.
   Deliberately excludes the horizontal component and the sweep-across-gates
   ordering — shed before skyscraper.
3. **The auto-encoder reconstructor**, behind the interface from (1), once there
   is a trained artifact and a decision about shipping weights in a
   GPL-licensed repository. May never be worth it, given per-system retraining
   plus the 1D blind spot.

## What the library will require of it

Small, concrete, and both are enforced by tests that fail rather than warn:

- **A figure shape must be declared.** `test_every_registered_filter_declares_a_figure_shape`
  asserts `figures.missing_shapes() == []`, so a new filter with no entry in
  `figures.SHAPES` breaks the suite. `CULLING` is the right shape, and the
  threshold panel applies if a threshold is recorded on the step report.
- **`docs/reference/` must be regenerated.**
  `test_the_committed_reference_matches_the_registry` compares the committed
  reference against the live registry, and `test_every_step_appears_in_the_reference`
  checks the filter is listed. The regeneration command is in the first test's
  docstring.

Beyond the tests, the agent index carries what a signature cannot — the
Ordering list above needs its two new entries, since that file is what another
agent reads to build a chain.

## Open questions

The profile criterion's own design questions are gathered in "Designing the profile criterion" above rather than listed here — including how it finds candidates without an infrastructure layer, and whether the sweep-across-gates ordering discriminates against a dipping conductor (**still the highest-value simulation to run**: one dipping plate, one power-line loop, same flight line, in SimPEG).

- **~~Do our deliveries carry raw stacks and measured uncertainties?~~**
  **Answered 2026-10-15.** SkyTEM carries a high-rate product and per-gate error
  estimates; VTEM sometimes carries a 10 Hz product and **never** error
  estimates. All of it is minimally processed by the vendor, not raw. So the
  modeled denominator is the default path. See "Two systems".
- **What do α and β_max become against a modeled sigma?** The remaining half of
  that question, and now the first thing to settle. A modeled sigma has
  different statistics from measured stack scatter, so 2.5 and 0.99 are not
  transferable on their face — and with vendor processing in the chain the
  calibration is per vendor and system. `parameter_sweep` against a manually
  processed line is the tool. **Needs real deliveries to do at all**, which is
  the next thing to get hold of.
- **How much does the vendor's pre-processing move the reconstruction
  residual?** Measurable directly once there is data: fit the basis, reconstruct
  known-clean soundings, and look at whether the residual distribution sits
  where a clean forward would put it. If the offset is uniform it only moves the
  threshold; if it varies along the line, the black box is doing something
  adaptive and the criterion is on shakier ground than it looks.
- **How is a survey's setting declared?** `corroborate` has to be set by
  someone who knows whether conductive targets are expected. A project-level
  fact, not derivable from the data, so it probably belongs beside the other
  project configuration rather than buried in a step.
- **Which PLNI derivation does each survey carry** — a tuned coil or a bandpass
  of the main receiver? It decides whether `corroborate: plni` is sound or
  actively harmful for that delivery, and it is not something the data
  announces. Worth asking each vendor once and recording per survey.
- **Which window** for the profile criterion — measured per crossing, or derived
  from flight height and line class? Measuring is probably better, with a
  derived value only bounding the search, because the realised extent runs well
  under the predicted one and varies pole to pole. See Appendix C.
- **Does line class need to be an input?** Voltage class proxies for tower
  height and span, and an infrastructure layer usually carries it. It would
  sharpen the window but makes the criterion depend on data we often lack.
- **Where a trained model lives.** Weights are not source, the repository is
  going public under GPL-3.0, and a model artifact carries its own provenance
  and licensing questions.
- **What does "working" mean?** There is no success criterion anywhere in this
  plan. "Asif's 96% is the number to beat" is not one: it does not say against
  whose judgement, on which survey, or counted how — per gate, per sounding, or
  per crossing. Per crossing is probably the honest unit, since missing a
  crossing entirely and trimming its edges differently are not comparable
  errors. **Decide this before building**, or the first result will be
  unarguable in both directions.

- **Validation, and the circularity in the obvious plan.** One real delivery, no
  labelled coupled data. Kang's released notebooks and SimPEG-EM — already in
  the stack as `YmerFlow/simpeg` — make synthetic coupled data with known ground
  truth reachable, and that is worth doing for controlled sweeps of flight
  height, loop count and ground-pathway resistivity.

  **But synthetic validation is partly circular and the plan should say so.**
  The coupling would be generated by a model whose weakest parameter — contact
  resistance, which Appendix C shows governs the whole circuit and which nobody
  can measure — is the one that decides the answer. The detector would then be
  scored on finding the coupling *that model* makes. The precedent is ours:
  `blossPowerlineContaminationAirborne2016` built a forward model that gave
  results "consistent with the collected gTEM data" but which "do not reproduce
  the amplitude or spatial-scale of powerline distortion observed in ATEM data."
  Kang's 3D treatment is better and still assumes negligible contact resistance,
  which is why their own field data came in at ~60% of prediction.

  So synthetic data can measure **sensitivity to parameters** and catch gross
  failures. It cannot establish that the detector finds real coupling. **The
  only validation that settles that is an expert marking up a real survey**, and
  that should be planned for rather than left as the thing SimPEG appears to
  replace.


---

# Appendix A — Why detection and not repair

Couplings to grounded infrastructure — powerlines, fences, pipes, rails — add a
response that is **coherent with the earth's**, because the transmitter induces
current in the conductor and the receiver sees the sum. Every source agrees on
the consequence:

> Couplings to such infrastructure influence the measured signal to a degree
> that cannot be corrected for, making it necessary to completely remove coupled
> data from the processed dataset.
> — `andersenArtificialNeuralNetworks2016`, citing Kirsch 2006

> Couplings to infrastructure cause the data to be corrupted to a degree that
> cannot be corrected. Therefore, the TEM data denoising approaches … are not
> feasible.
> — `asifAutomatedTransientElectromagnetic2022`

**It is also not a powerline harmonic problem.** `larsenRemovalPowerlineNoise2022`
removes 50/60 Hz harmonics from fully-sampled time series, and notes that TEM
already suppresses those at acquisition, by synchronous detection with
alternating transmitter polarity. Our data arrives stacked, so the harmonics are
gone before we see it. What remains is coupling — a different phenomenon with a
different signature.

## Stripping: why not, and the one paper that did it

Forward-modeling the coupling and subtracting it is ruled out here, but the
reason is narrower than "it cannot be done" — because it has been done.

**`polzerStrippingCulturalConductor1990` strips cultural response successfully,
without modeling the grounding at all:**

> we do not have to model the response of a grounded conductor to remove most of
> its effect from the data. This is because most of the vertical magnetic field
> response arises only from the induced currents flowing within the conductor
> itself which need only be estimated by measurement.

At each delay time the method decomposes the measurement along a short traverse
across the conductor into the field of a **unit-ampere current element**
following the conductor's known path, scaled by an unknown current, plus a
**smooth quadratic background** standing for the earth, and solves least squares
for the current. Subtracting the fitted conductor term leaves stripped data. It
works, on UTEM data over powerlines and fences, and it is honest about the
residual risk.

**Three things stop it transferring to production airborne work:**

1. It needs **detailed knowledge of the 3D path of each conductor**. Polzer
   notes skywires are the hard case — "it is impossible to precisely survey the
   location and height of the skywire, particularly as its height must follow a
   catenary curve between suspension points."
2. It needs **dedicated short traverses across the conductor** to estimate the
   current. A production flight line crosses a powerline once, at whatever angle
   the survey design gives it.
3. Because the conductor is grounded at many points the current is not constant
   along it, so it must be treated as "a number of non-overlapping pieces, each
   with its own traverse profile" — multiplying the traverses needed.

So stripping is a **survey-design method**, not a processing method.

The modeling route was tried and did not reach ATEM scale.
`blossPowerlineContaminationAirborne2016` built a forward algorithm around a
'skywire' RL analogue, after Polzer, with results "consistent with the collected
gTEM data" but which "do not reproduce the amplitude or spatial-scale of
powerline distortion observed in ATEM data." Parameterising it properly needs
the **contact resistance at each pole** and the **capacitance between the ground
wire and ground**, neither knowable per survey. Every crossing is different;
where a pole's grounding is poor there may be little response at all, so even a
calibrated model would not transfer between two crossings of the same line.

**What Polzer gives us anyway.** His decomposition is the ancestor of the
profile criterion: Polzer fits *smooth background + conductor response* along a
traverse in order to **subtract**; the profile criterion looks for *smooth
background + symmetric departure* along a flight line in order to **detect**.
Detection is the weaker ask, and weaker in exactly the places Polzer is
expensive — no conductor geometry, no unit-current template, no dedicated
traverse. His result that the stripped response is "symmetric about the loop
with a positive amplitude envelope … of about 40%" independently corroborates
the symmetry the 2016 abstract describes.

---

# Appendix B — The oscillatory signature, and why it is usually geology

A late-time sign reversal has at least three causes, and the library's own
specification already says so — `01`, quoted in `filter_32`'s docstring:

> a sign change indicates coupling, **an IP effect**, or noise dominating

`cull_negatives` culls all three identically. One of them is geology.

**Induced polarization is not a special case.** Any interface that charges
produces it: a conductive clay layer in a sand or sandstone background polarizes
at the contact, which is an ordinary sedimentary situation rather than an exotic
one. In mineralized ground, sulphides are strongly polarizable and effectively
glow in the data. And IP is present long before it is strong enough to drive the
transient negative — a negative is the end of a continuum, not the start of one.

It is also **recoverable information, not just a nuisance**:
`macnaeQuantifyingAirborneInduced2016` and
`macnaeQuantitativeEstimationIntrinsic2016` quantify and estimate intrinsic IP
parameters from airborne EM, and `grombacherInducedPolarizationEffects2021`
documents the effects in real ATEM data. Culling a negative is therefore not a
neutral act of hygiene — it discards a measurement somebody else publishes
methods to invert.

**"Capacitive" is the wrong label for most of what we see, and it matters which
sense is meant.** `kangEffectPowerLines2021` uses the term for a specific
infrastructure mechanism — a buried insulated cable acting as a capacitor — and
concludes within that scope that "without large metallic objects such as pipes,
it will not be possible to observe significant capacitive coupling in AEM data."
That is a statement about the well-associated noise sources that paper examines.
It does not cover the late-time oscillatory behavior we most often meet, which
is polarizable geology and needs no infrastructure at all. When this plan says
the oscillatory signature is usually signal, it means **IP**, not that mechanism.

## Culling IP is deliberate, and should be said out loud

None of the above is an argument that `cull_negatives` is wrong. **We cannot
model IP at present**, and data the forward model cannot represent cannot be
inverted correctly — so culling it is the honest response, and it is what the
industry does. The same applies to three-dimensionality: everything here is 1D,
a known flaw and also standard practice.

What is worth changing is not the behavior but the **account of it**. The
library should say that it is discarding signal and why, rather than implying a
negative is merely noise. `filter_32`'s defence of `disable_tail` — "a transient
does not recover once it has gone bad" — is true of coupling and of noise, and
false of IP, where the transient is doing something physical and repeatable.
Culling it is a decision forced by the inversion's assumptions, not a judgement
about the data.

Recording it that way costs nothing and matters later: when an IP-capable
forward model arrives, the thing to revisit is findable, and anyone reading a
culled dataset in the meantime knows what went and why.

The inverse limit is Asif's, and bounds what a good reconstruction score means:
"when a coupling response mimics a geological response. If such data can be
represented by 1-D model parameters, the autoencoder reconstructs the data with
low error and the data will not be discarded." The criterion is a test of
one-dimensionality, not of cleanliness.

---

# Appendix C — The physics of the coupling

`kangEffectPowerLines2021` simulates the mechanism in 3D — the copper wire
between poles, the wire down each pole, the ground electrodes, and the **ground
pathway** closing the circuit — and cites
`blossPowerlineContaminationTimedomain2017` for the circuit treatment of ground
contact resistance. It supplies the two things the profile criterion was
missing: a reason to trust the central trough, and a way to size the window.

## Why the centre is clean

Not an artifact, and not a coincidence:

> When AEM loops are right above the power line, there will be no magnetic flux
> (generated by induced currents flowing in the power-line loop) intersecting
> the receiver loop; as a result, there is no power-line effect. As the AEM
> loops move away from the power line, the amount of magnetic flux intersecting
> the receiver loop increases, reaching a peak when the coupling is maximized,
> and then it decreases as the loops move farther away.

Flux cancellation directly over the loop. The double peak with a null between is
**what the geometry requires** — which is also why geology can produce it, since
a steeply dipping conductor is the same geometry (see "The double peak is not
diagnostic").

## The window, and why it is not a constant

**Numbers — but read the geometry first.** These hold for *one* power line:

| Kang's modeled line (Table 3) | |
|---|---|
| pole height | **10 m** |
| span, i.e. one loop's length | **80 m** |
| loops simulated | 15, total 1.2 km of line |
| transmitter height | 40 m |
| half-space | ~20 Ω·m |

| result for that geometry | |
|---|---|
| peak effect | TPPE ≈ 44%, at **~30 m** from the line |
| below 3% (their noise level) | beyond **174 m** |
| observed in their field data | high-quality soundings from only **~60%** of that |

**10 m poles on 80 m spans is a distribution line**, not a transmission line.
Kang state the scaling: "the area of the power-line loop is proportional to the
moment of the power-line loop, which is equivalent to the product of the current
and the area of the power-line loop." Loop area is **span × pole height**, so a
transmission line on 40 m towers at 350 m spans encloses roughly seventeen times
the area simulated — though area is not the whole story, because inductance
rises with loop size and limits the current (below).

`blossPowerlineContaminationTimedomain2017` Fig. 2-6 separates the classes for
exactly this reason — transmission above 230 kV, subtransmission around 100 kV,
distribution below 69 kV, residential under 1000 V — and shows different
signatures for a transmission crossing (Fig. 2-7) and a two-wire distribution
crossing (Figs. 2-9, 2-10).

**So 174 m is not a constant to code against.** It is one point on a surface.
The 300 m and ~1 km figures elsewhere are plausibly the same physics over larger
loops rather than a contradiction, but that should be checked against which line
type each observation came from before anyone relies on it.

**Kang's figure is an upper limit in three ways**, two of which they state:

1. **Negligible ground contact resistance** is assumed. Their field data shows
   clean soundings from only ~60% of the predicted distance, attributed to "the
   high ground contact resistance caused by degradation of the grounded
   electrodes" — 174 m predicted against ~104 m observed.
2. **No catenary.** "This catenary effect reduces the area of the power-line
   loop; thus, the simulated power-line effect could be larger than the true
   power-line effect." Sag grows with span, so the correction grows with exactly
   the parameter that makes transmission lines larger.
3. **A small loop.** All of it applies to an 800 m² loop.

**What the detector should take from this is a dependence, not a distance:**

| grows with | shrinks with |
|---|---|
| span × pole height (loop area) | transmitter height |
| number of contributing loops, to ~15 | **total loop resistance**, i.e. contact resistance (unknowable) |
| | loop inductance, which rises with loop size |
| | catenary sag |

Note what is absent: the bulk resistivity of the ground. The search width should
come from flight height and **line class where known** — an infrastructure layer
usually carries voltage class, which proxies for tower height and span. Where it
is not known, the extent should be **measured per crossing**, which is the safer
default anyway given that the realised value runs under the predicted one and
varies pole to pole.

## What limits the loop current is contact resistance, not the background

The detector needs a prior about where coupling is likely, and the obvious one —
that resistive terrain is safer — does not hold up.

Treat the skywire as a circuit, as
`blossPowerlineContaminationTimedomain2017` §2.2.4.2 does: six resistive
elements — the top wire, two pole wires, the earth return, and two contact
resistances at the grounding points. The conductor terms are trivial. The earth
return is the interesting one, and it is easy to get backwards.

Current between two grounded poles does not run along a narrow corridor; it
spreads into the half-space as a curved bundle of flow lines — the current
banana. Borrowing the pole-pole array from DC resistivity, the transfer
resistance between two surface electrodes separated by `a` is

```
R_earth = ρ / (2 π a)
```

**inversely** proportional to separation, so longer spans have *lower*
earth-return resistance.

That is not a quirk of the pole-pole array. It is the general result the
electrical and MT literature states in plainer form: **more volume engaged means
more parallel current pathways means lower resistance.** Widening the separation
drives current deeper and wider, recruits more of the half-space, and the
resistance falls. Any intuition built on "longer path, more resistance" — which
is what a fixed-cross-section model encodes — has the sign wrong.

The thesis works an example: a 100 m span on a 100 Ω·m half-space gives
**0.16 Ω**, against 4.8 mΩ for 150 m of conductor. Take the case intuition says is
protective — **1000 Ω·m with poles 30–40 m apart — and it is still only 5.3 to
4.0 Ω.** Single digits on ground an order of magnitude more resistive than most
survey areas.

So the earth return is almost never the limiting term, and the thesis says what
is: "Neglecting the CR at the grounding points the total resistance of the
Skywire is 0.16 Ω! This low number suggests that the CR is the limiting factor
here." **Contact resistance dominates the circuit** — aimed at <20 Ω, degrading
unpredictably with corrosion, construction and season — and it is the one
quantity no survey can measure.

Three things follow:

1. **Do not deprioritise resistive terrain.** Bulk resistivity enters the loop
   only weakly, so a 300+ Ω·m survey area is not thereby safe. The *percentage*
   effect moves the other way as well: Kang note that "an increase in
   ρ_background causes a decrease in the amplitude of the subsurface signals,
   resulting in a net increase in the percentage power-line effect." Resistive
   ground may be the harder case.
2. **The threshold that matters is a circuit threshold.** The thesis found one
   by varying loop resistance directly: "Resistances above 100 ohms yield
   Skywire contributions so low in amplitude that when superimposed on the
   geologic response the data were largely unchanged." That is total loop
   resistance, overwhelmingly contact resistance — and it fits Kang's field
   observation about electrode degradation. Both say the grounding decides.
3. **The snowflake problem, restated as a circuit.** The quantity that decides
   whether a crossing is loud or silent cannot be known, so the detector must
   carry no prior about which surveys or crossings will be affected.

**This also tempers the transmission-versus-distribution scaling.** Loop area
raises the EMF, but inductance rises with loop size and limits the resulting
current: the thesis gives **0.67 mH** for a 225 m × 20 m transmission-style
skywire against **0.14 mH** for a 100 m × 10 m one, and finds that "as inductance
increases, the maximum amplitude of the Skywire response lessens", with the
response pushed later in time. So bigger lines are not simply "17× worse" — this
plan said that too confidently once. The search *window* should still scale with
loop size; the amplitude does not scale with it the same way.

## LM or HM? The effect lives in a time window, not in a moment

**Field experience is that the LM sees less powerline effect than the HM.** The
simulation result in `kangEffectPowerLines2021` is LM-stronger for the case
modeled there — "the LM response shows a stronger double-peak signature than the
HM response … due to the limited frequency spectrum of the power-line effect,
which is mostly included in the LM response" (E131) — and the paper also notes
the ordering reverses under other conditions, so HM-dominance is within what it
describes rather than against it. The reconciliation is useful to the design.

**1. The controlling variable is where the loop puts its energy in time.** Lower
loop resistance lengthens the time constant and pushes the response later;
higher resistance pulls it earlier. Kang see this in ρ_g (E131, and Figure 9,
where the late-time HM effect grows steadily as ρ_near falls from 340 to 1 Ω·m),
and `blossPowerlineContaminationTimedomain2017` §3.1.2.4 sees the same thing in
a different code by varying loop resistance directly:

> As Skywire resistance increases, the observed anomaly is earlier in time and
> as Skywire resistance decreases the observed anomaly is pushed to later times.

Two independent codes, same dependence. So whichever moment's gates sit over
that time window is the one that looks worst — and since loop resistance is
dominated by contact resistance, which varies pole to pole, so does the answer.

**2. Kang's field data shows the effect peaking between the two moments.** From
E134, on real SkyTEM 312 data:

> (1) the magnitude of the power-line effect in the LM data **increases with
> increasing time channel**, and (2) the power-line effect in the HM data **is
> strong in the early time channels, but it decreases with the increasing time
> channels**.

LM spans 0.016–1.4 ms and HM spans 0.36–11 ms, so both trends point at the same
place: the effect is maximal in the **overlap window, roughly 0.4–1.4 ms**, and
falls away on both sides. Late LM gates and early HM gates are the same times.

**The powerline effect peaks in a time window, not in a moment.** Which moment
appears worse depends on where that window sits relative to each moment's gates
and how large the earth response is there — all survey-dependent.

**3. And Kang identify a mechanism that suppresses the LM in the field.** They
note their measured LM effect comes in below the simulated one, and say why:

> we expect and see the first few time channels of the LM data show a power-line
> effect smaller in magnitude than the simulation results; this is due to the
> increasing flight height around the power line.

**Pilots climb over powerlines.** They also show (E132) that "the LM response
shows greater sensitivity to changes in h than does the HM response," because
early gates decay geometrically faster with height. The two-way dipole path goes
as 1/r³ down and 1/r³ back, so 1/r⁶ overall — a 20 m climb on a 40 m nominal is
roughly a factor of 11, larger than most of the effects discussed here. That is
the paper's own explanation of why field LM is quieter than modeled LM, and it
generalises.

**What the detector should do:**

- **Run on all moments, and privilege none.** On a dual-moment system that is
  two passes; on single-moment VTEM the question does not arise.
- **Expect the strongest signature near the LM/HM crossover**, and treat the
  time axis as part of the pattern rather than collapsing it.
- **Treat a height excursion as a corroborating covariate** — see Corroboration.
- **Do not calibrate an amplitude threshold on simulated LM data.** It will be
  optimistic by whatever the pilot did.

**And TPPE is a ratio** — worth remembering when comparing anything to anything.
Kang's 44% is a percentage of the subsurface response, so a moment can look
worse simply by having a weaker earth signal at the times it measures. Absolute
and percentage comparisons are different questions, and the one that matters for
inversion damage is the percentage.

**And it may solve the validation problem.** Kang released the Jupyter notebooks
that generated the paper's examples, and the 3D simulation code is SimPEG-EM,
already in our stack as `YmerFlow/simpeg`. **Synthetic coupled data with known
ground truth is reachable** — the labelled set we otherwise do not have.

---

# Appendix D — Sources

All keys are in `~/Codes/_refs/library.bib`; PDFs in the YmerFlow shared library.

- `asifAutomatedTransientElectromagnetic2022` — the auto-encoder expert system,
  and the single most useful paper here. Section V is the one to reread: the
  four tunable parameters and what they mean, the raw-stack requirement,
  retraining for other systems, and the authors' own statements of the 1D/3D and
  mineral-exploration limits
- `kangEffectPowerLines2021` — 3D simulation of the mechanism: the
  flux-cancellation explanation of the trough, the distance and TPPE numbers,
  the LM/HM and transmitter-height sensitivities, field validation against 41
  water wells, and released notebooks that make synthetic coupled data reachable
- `blossPowerlineContaminationTimedomain2017` — the signatures, the profile
  characterisation, the skywire circuit treatment (§2.2.4.2) and the resistance
  and inductance experiments (§3.1.2.4)
- `blossPowerlineContaminationAirborne2016` — the quoted profile
  characterisation and the smearing failure
- `andersenArtificialNeuralNetworks2016` — the earlier supervised ANN
- `polzerStrippingCulturalConductor1990` — stripping by measured current; the
  ancestor of the profile decomposition
- `xieImprovedPrincipalComponent2014` — PCA denoising; a reconstruction basis
- `larsenRemovalPowerlineNoise2022` — harmonic removal; **not applicable here**,
  but its discussion of remote-reference cancellation is the closest thing in
  the library to an MT treatment, and describes what the powerline monitor is
- `macnaeQuantifyingAirborneInduced2016`, `macnaeQuantitativeEstimationIntrinsic2016`
  — airborne IP quantified and estimated: the oscillatory signature is
  recoverable signal, not only a nuisance
- `grombacherInducedPolarizationEffects2021` — IP effects in real ATEM data
- Yin & Wang 2015, powerline ground resistance from helicopter EM. **Not read**

**Not in the library:** no magnetotelluric or self-potential papers. The
volume/pathways result in Appendix C is standard in that literature but is
stated here without a citation; worth sourcing properly if the argument ever
leaves this document.
