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

Four limits. The first two are deliberate, the last two are the design risk.

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
   a setting, not a constant.
2. **Neither criterion may run unsupervised on mineral data.** Belongs in the
   docstring, not only here.
3. **The second pass is not optional there.** A tool that can discard the target
   needs a human reading what it discarded.

This does not sink the approach. Most surveys in hand are hydrogeological or
sedimentary, where a bedded clay layer will not imitate a vertical sheet. But
the detector has to be told which setting it is in; it cannot work that out.

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

### Flag or cull is two registrations, not a parameter

The switch is wanted, and the library already has it — but as its central
split rather than as an `action` argument. `core/qc/__init__.py`:

> **QC writes attributes. Culling writes the mask.** … A check writes its Good /
> Marginal / Out of Spec verdict as a column on `flightlines`, which changes
> nothing about what the inversion sees. Culling writes the in-use mask, which
> does. QC records a judgment; culling acts on one.

An `action: flag | cull` parameter would straddle that boundary, and the two
behaviors already have different entry-point groups and different runners. So:

| | group | writes | |
|---|---|---|---|
| `coupling_detected` | `ymerflow.processing.qc` | a **proposed mask**, plus a per-sounding confidence | the flag |
| `cull_by_coupling` | `ymerflow.processing.filters` | the real mask and the provenance | the cull |

**Two registrations over one detection core.** The reconstruction and profile
criteria, the confidence metric, the smoothing and the corroboration check are
shared; only the last act differs. You choose by which step goes in the chain,
and a chain may hold **both** — QC first, so the proposal is recorded whatever
happens next, then the filter if this survey wants the cull. That is strictly
better than a parameter, because the proposal survives into the output even on
a culling run, which is what pass two needs to reverse anything.

#### The flag has to be per gate, which the verdict column cannot do

Every QC check so far judges **auxiliary data** — altimeter, tilt, speed,
current — single channels that describe a whole sounding, so a verdict column
on `flightlines` fits exactly. **Coupling is not like that.** It affects a band
of gates, the band moves with distance from the line, and the whole
time-window finding (Appendix C) says *which* gates is the information. One
value per sounding throws that away.

So the QC side writes a **proposed mask** into `layer_data` — same shape and
same encoding as `inuse_{channel}`, under a **prefixed** key so it stays
outside the anchored channel-discovery patterns, which is the lesson `why_`
already encodes. Five things fall out of reusing the mask's own shape:

1. **It is per gate**, which is what the phenomenon is.
2. `culled_fraction` arithmetic works by pointing at a different array.
3. The figure code in `figures.py` already draws culling; it needs no new shape.
4. Promoting a flag run to a cull run is a copy.
5. **It is the same object the frontend's edit format already carries** — see
   below, which is the part worth building around.

The per-sounding confidence column stays, but with a different job: it is the
**index**, not the payload. It drives the per-line and per-flight rollups in
`aggregate.py`, and it is what somebody scans to find which crossings to look
at. The proposed mask is what they then look at.

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

A flag is a third category of the same kind — proposed by the algorithm rather
than by a person — so it is one more state in a shader that already branches on
four, fed by an array shaped exactly like one it already reads.

By contrast, **nothing in the frontend reads `qs_` columns at all** — a search
returns nothing, for any QC check. So the per-sounding verdict route, which
looked like the cheap one, has no rendering whatsoever, while the per-gate route
is mostly there. That is the opposite of the intuition and it decides the design.

#### Which closes the loop with the manual-edit machinery

The frontend's edit format is `layer_data["InUse_{channel}"]` with **NaN meaning
no override**, addressed by `flightlines.apply_idx`, marked by
`model_info.diff_dummy`. **A proposed mask is that object**, with 0 where the
detector proposes culling and NaN everywhere else.

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
    criterion: reconstruction     # reconstruction | profile | both
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
infrastructure: ...      # vector geometry, as filter_30 takes it
```

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

## Staging

0. **The QC check.** `coupling_detected` — the detection core, the per-sounding
   confidence, and a **proposed mask** per gate; the real mask untouched. The
   smaller half of the work, useful on its own as a QA/QC product, and it keeps
   the culling step honest by existing separately. Emitting the proposal in the
   existing diff format makes it reviewable in the editor that already exists;
   the front-end work is a path to load a pipeline-emitted diff, plus one more
   state in `ChannelPlot`'s overlay.
1. **The framework, with a linear reconstructor.** Residual in sigmas, the
   confidence metric and column, along-line smoothing, the fuzzy decision, mask
   and provenance. No new dependency. Testable against synthetic couplings now.
   Fitted on the delivered survey, it also absorbs the vendor's pre-processing
   instead of mistaking it for an anomaly — see "The black box is a systematic".
2. **The profile criterion** — the double-peak-with-central-trough search. The
   part that addresses the real gap, works on both systems and on delivered
   data, and has nothing to copy.
3. **The auto-encoder reconstructor**, behind the interface from (1), once there
   is a trained artifact and a decision about shipping weights in a
   GPL-licensed repository. May never be worth it, given per-system retraining
   plus the 1D blind spot.

## Open questions

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
- **Does the sweep-across-gates ordering survive as a discriminator?** The one
  candidate left that is geophysical rather than circumstantial: does a steeply
  dipping conductor, approached along a flight line, produce the same ordered
  progression through the gates — late first on approach, early first on
  departure? If not, the profile criterion has a real discriminator and the
  mineral restriction loosens. **The highest-value simulation to run**, and
  SimPEG can run it: one dipping plate, one power-line loop, same flight line.
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
- **How the profile criterion finds candidates** without being told where the
  powerlines are. With an infrastructure layer it is easy; without, it is an
  unsupervised search for a recurring symmetric pattern.
- **Where a trained model lives.** Weights are not source, the repository is
  going public under GPL-3.0, and a model artifact carries its own provenance
  and licensing questions.
- **Validation.** One real delivery, no labelled coupled data. Kang's released
  notebooks and SimPEG-EM — already in the stack as `YmerFlow/simpeg` — make
  **synthetic coupled data with known ground truth** reachable: vary
  ground-pathway resistivity, flight height and loop count, and measure what the
  detector catches and misses. It does not replace agreement with an expert's
  manual culling on real data (Asif's 96% is the number to beat), but it is the
  only way to measure the false-negative rate at all.

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
