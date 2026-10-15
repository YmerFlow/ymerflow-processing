# Detecting coupled data

**Status:** planned, not started. Literature reviewed 2026-10-15.
**Decided:** detect and cull only, never repair. The powerline monitor
corroborates; it does not trigger. Primarily a **QA/QC tool**, run unattended on
a first pass and adjusted by hand on a second. Both **VTEM and SkyTEM** are in
scope.

## The problem, stated by the sources

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

So this is a **detection** problem, not a repair problem.

It is also not a *powerline harmonic* problem. `larsenRemovalPowerlineNoise2022`
removes 50/60 Hz harmonics and their integer multiples from fully-sampled time
series, and notes that TEM already suppresses those at acquisition, by
synchronous detection with alternating transmitter polarity. Our data arrives
stacked, so the harmonics are gone before we see it. What remains is coupling —
a different phenomenon with a different signature.

## Two signatures, and we catch only one

`blossPowerlineContaminationTimedomain2017` Fig. 1-1 separates them by shape:

| | what it looks like |
|---|---|
| "capacitive" | oscillatory behavior at later times |
| inductive / galvanic | **smoothly elevated** dB/dt curves |

The existing culling filters — `cull_negatives`, `cull_by_curvature`,
`cull_by_decay_rate` — are shape tests on one sounding at a time. They catch the
oscillatory kind, because oscillation breaks monotonicity and curvature.

**They cannot catch smooth elevation.** A galvanically coupled sounding is still
a plausible-looking decay; it is simply in the wrong place. Nothing about its
shape is wrong.

### But the oscillatory half is not one thing, and is often signal

A late-time sign reversal has at least three causes, and the library's own
specification already says so — `01`, quoted in `filter_32`'s docstring:

> a sign change indicates coupling, **an IP effect**, or noise dominating

`cull_negatives` culls all three identically. One of them is **geology**.

**Induced polarization is not a special case.** Any interface that charges
produces it: a conductive clay layer in a sand or sandstone background
polarizes at the contact, which is an ordinary sedimentary situation rather
than an exotic one. In mineralized ground, sulphides are strongly polarizable
and effectively glow in the data. And IP is present long before it is strong
enough to drive the transient negative — a negative is the end of a continuum,
not the start of one.

It is also **recoverable information, not just a nuisance**:
`macnaeQuantifyingAirborneInduced2016` and
`macnaeQuantitativeEstimationIntrinsic2016` quantify and estimate intrinsic IP
parameters from airborne EM, and `grombacherInducedPolarizationEffects2021`
documents the effects in real ATEM data. Culling a negative is therefore not a
neutral act of hygiene — it discards a measurement somebody else publishes
methods to invert.

**So `kangEffectPowerLines2021` is too narrow on this point.** It treats
capacitive effects as arising from a buried insulated cable acting as a
capacitor, and concludes that "without large metallic objects such as pipes, it
will not be possible to observe significant capacitive coupling in AEM data."
That is sound for the noise sources *that paper* set out to examine — those
associated with water wells — but it is not the whole account of late-time
oscillatory behavior in AEM, because polarizable geology produces it without
any infrastructure present at all.

### What this means for scope

**This detector targets smooth elevation only.** Sign reversals and late-time
oscillation are explicitly out of scope, because the reading of them is
ambiguous and one reading is signal. Elevation is the unambiguous case: no
geology produces a smooth, symmetric, locally confined lift in dB/dt about a
mapped conductor with a null directly over it.

That narrows the filter and makes it more defensible.

### Culling IP is deliberate, and should be said out loud

The above is not an argument that `cull_negatives` is wrong. **We cannot model
IP at present**, and data the forward model cannot represent cannot be inverted
correctly — so culling it is the honest response, and it is what the industry
does. The same applies to three-dimensionality: everything here is 1D, which is
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

### The 1D assumption is a false-positive source for *this* detector

This is the part that bears directly on the design rather than on documentation.

A reconstruction-residual detector flags whatever its basis cannot reproduce. If
the basis is built from 1D responses, then "implausible" means "not 1D" — and
**genuine three-dimensional geology will produce reconstruction error**.

Asif's auto-encoder has exactly this blind spot by construction: it is trained
on a million forward responses from **1D** resistivity models, so a compact 3D
conductor is as foreign to it as a powerline. They do not test the case, but
they do state it plainly:

> If there are 2-D or 3-D effects in the data that cannot be represented by any
> 1-D resistivity model, a low confidence value may be obtained for such data,
> which consequently might be identified as an anomaly by our expert system.

Their proposed remedy is to build a 2D/3D model database, which they
acknowledge "would be much more computationally expensive." It is not available.

The consequence is pointed: a detector of this kind can cull the anomaly the
survey was flown to find. That is a worse failure than missing a coupling,
because a missed coupling produces a visible artifact somebody can question,
while a culled target leaves nothing behind to notice.

**And it gets sharper outside hydrogeology.** Asif scope their work explicitly:

> Our methodology is designed in a hydro-geological context. If the
> infrastructure that induces coupling response in the data is considered as
> exploration targets, a different strategy may be required where similar
> coupling patterns are identified. However, it is beyond the scope of this work.

A mineral survey is the case they are excluding. A compact, highly conductive,
strongly polarizable body is both the thing the survey is for and the thing most
likely to reconstruct badly against a 1D basis. **The reconstruction criterion
should not be run unsupervised on mineral exploration data**, and that belongs
in the docstring rather than in a plan nobody rereads.

The inverse failure is theirs too, and limits what a good score means: "when a
coupling response mimics a geological response. If such data can be represented
by 1-D model parameters, the autoencoder reconstructs the data with low error
and the data will not be discarded." The criterion is a test of
one-dimensionality, not of cleanliness — which is the honest description of what
it measures, and worth keeping in those words.

Two things follow:

- **The profile criterion is less exposed to this than the reconstruction
  criterion.** It keys on a geometric signature — symmetric, locally confined,
  null in the centre — rather than on fidelity to a 1D model. A 3D conductor
  produces an anomaly, but not generally a symmetric double peak with a null
  directly over it.
- **Any reconstruction criterion needs a 3D false-positive test** before it
  culls anything in anger, and Kang's released SimPEG-EM makes one buildable:
  simulate a compact 3D conductor with no powerline present and confirm the
  detector leaves it alone.

That is not inferred. `blossPowerlineContaminationAirborne2016` states it:

> When looking at one sounding (dB/dt vs. time) by itself, **it is nearly
> impossible to tell that the data are affected**; however, when looking at a
> profile of data (dB/dt vs. distance) the effect is quite prominent as a
> **symmetric double peak signature, extending up 300 m away from the powerline,
> and with soundings directly above the powerline largely undistorted.**

And the cost of missing it, from the same source: uninverted, coupled data
yields "artificial conductors in the subsurface, and if inverted with a
laterally- or spatially-constrained inversion, these false conductors get
smeared out into the model domain."

## Why the published automation does not close that gap

Both machine-learning approaches judge **one sounding, or one raw stack, at a
time**:

- `andersenArtificialNeuralNetworks2016` — a supervised ANN trained on manually
  labelled field data. Asif criticises this fairly: manual labels make the
  method "biased to local survey areas and limited geological conditions", and
  carry "subjective inconsistent human labeling".
- `asifAutomatedTransientElectromagnetic2022` — a convolutional auto-encoder
  trained on **synthetic** uncoupled responses, so no labels are needed.
  Coupling is found as an **anomaly**: reconstruct the sounding, and where
  reconstruction fails, something is present that the space of plausible earth
  responses does not contain. 96% agreement with an expert processor.

The auto-encoder is the better of the two and is worth having. But it is a
per-sounding test, and Asif state its failure mode in exactly our terms:

> a buried power line excited by the tTEM transmitter … the impact of this
> coupling appears as an **enhanced signal level, which could easily be
> interpreted as a geological change**. The manual operator has removed all data
> in the vicinity of the road … The automated approach cannot distinguish this
> type of coupling from uncoupled data.

**That is the smooth inductive case.** The published state of the art fails on
precisely the half our shape filters also miss.

## The part nobody has used

The profile signature is characterised, and unexploited by any automated method
in the library:

- **symmetric**, with a **double peak**
- a **central trough** — the sounding over the line is largely undistorted
- extent ~300 m from the line (2016 abstract); ~1 km wide at late times
  (2017 Fig. 2-4). **Resolve this discrepancy before choosing a window.** It may
  be per line type, or per time gate — Fig. 2-8 notes the affected region is
  narrower at early times than late.
- wider at late time than early (Fig. 2-8)

To which Asif add a fourth dimension the thesis figures do not isolate — the
pattern **sweeps across the gates** as the system passes:

> the signature of couplings generally appears in late time data points first
> and in early time data points afterward as the system approaches the coupling
> source. On the contrary, when the system moves away from the coupling source,
> the coupling effect fades in early time data points first and in late time
> data points afterward.

That is a signature in the time–distance plane, not just in distance, and it is
ordered: late gates light up first on approach and go out last on departure.
Geology crossing a flight line does not do that. It is probably the strongest
single discriminator available to the profile criterion, and it is free — the
data is already a 2D array of gate against distance.

A per-sounding detector run across this pattern does the wrong thing twice: it
flags the flanks and **keeps the centre**, leaving an island of apparently good
data directly over the conductor.

Detecting the pattern rather than the sounding is the contribution available
here, and it is ours.

## The trough is physics, and the window is derivable

`kangEffectPowerLines2021` simulates the mechanism in 3D — the copper wire
between poles, the wire down each pole, the ground electrodes, and the **ground
pathway** closing the circuit — and cites `blossPowerlineContaminationTimedomain2017`
for the circuit treatment of ground contact resistance. It supplies the two
things the profile criterion was missing: a reason to trust the central trough,
and a way to size the window instead of guessing it.

**Why the centre is clean.** Not an artifact, and not a coincidence:

> When AEM loops are right above the power line, there will be no magnetic flux
> (generated by induced currents flowing in the power-line loop) intersecting
> the receiver loop; as a result, there is no power-line effect. As the AEM
> loops move away from the power line, the amount of magnetic flux intersecting
> the receiver loop increases, reaching a peak when the coupling is maximized,
> and then it decreases as the loops move farther away.

Flux cancellation directly over the loop. The double peak with a null between is
**what the geometry requires**, which is what makes it worth detecting rather
than a pattern we happened to notice.

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

**10 m poles on 80 m spans is a distribution line.** It is not a transmission
line, and the difference is not small. Kang state the scaling themselves: "the
area of the power-line loop is proportional to the moment of the power-line
loop, which is equivalent to the product of the current and the area of the
power-line loop." The loop area is **span × pole height**, so a transmission
line on 40 m towers at 350 m spans encloses roughly seventeen times the area of
the case simulated.

`blossPowerlineContaminationTimedomain2017` Fig. 2-6 separates the classes for
exactly this reason — transmission above 230 kV, subtransmission around 100 kV,
distribution below 69 kV, residential under 1000 V — and the thesis shows
different signatures for a transmission crossing (Fig. 2-7) and a two-wire
distribution crossing (Figs. 2-9, 2-10).

**So 174 m is not a constant to code against.** It is one point on a surface,
and the 300 m and ~1 km figures elsewhere are plausibly the same physics over
larger loops rather than a contradiction — though that should be checked
against which line type each observation came from before anyone relies on it.

**Kang's figure is an upper limit in three separate ways**, two of which they
state:

1. **Negligible ground contact resistance** is assumed. Their own field data
   shows clean soundings from only ~60% of the predicted distance, which they
   attribute to "the high ground contact resistance caused by degradation of
   the grounded electrodes." This is the snowflake problem with a number on it:
   174 m predicted, ~104 m observed, the gap being corroded grounding that
   varies pole to pole.
2. **No catenary.** "The power line is assumed to be straight; thus, the
   catenary effect related to the curvature in the power line between the poles
   is not taken into account. This catenary effect reduces the area of the
   power-line loop; thus, the simulated power-line effect could be larger than
   the true power-line effect." Sag reduces the enclosed area, and the longer
   the span the more it sags — so the correction grows with exactly the
   parameter that makes transmission lines worse.
3. **A small loop.** Everything above applies to an 800 m² loop.

**What the detector should take from this.** Not a distance. A dependence:

| grows with | shrinks with |
|---|---|
| span × pole height (loop area) | transmitter height |
| conductivity of the ground pathway | ground contact resistance (unknowable) |
| number of contributing loops, to ~15 | catenary sag |

The search width should come from the survey's own flight height, its average
resistivity, and **the line class where that is known** — an infrastructure
layer usually carries voltage class, which proxies for tower height and span.
Where it is not known, the extent has to be **measured per crossing** rather
than assumed, which is the safer default anyway given that the realised value
runs well under the predicted one and varies pole to pole.

**Two further consequences for the detector:**

- **The low moment carries the signature more strongly.** "the LM response shows
  a stronger double-peak signature than the HM response … due to the limited
  frequency spectrum of the power-line effect, which is mostly included in the
  LM response." As ρ_g falls, lower-frequency content grows and the effect
  reaches later gates. So LM is the better place to look first, which is the
  opposite of Asif's tTEM choice to train on HM.
- **Conductive ground makes it worse**, and also makes the earth response
  smaller — "an increase in ρ_background causes a decrease in the amplitude of
  the subsurface signals, resulting in a net increase in the percentage
  power-line effect." A fixed amplitude threshold would therefore be wrong in
  both directions across a survey.

**And it may solve the validation problem.** Kang released the Jupyter notebooks
that generated the paper's examples, and the 3D simulation code is SimPEG-EM —
already in our stack as `YmerFlow/simpeg`. That means **synthetic coupled data
with known ground truth is reachable**, which is the labelled set we otherwise
do not have. See the open questions.

## Stripping: why not, and the one paper that did it

Forward-modeling the coupling and subtracting it is ruled out here, but the
reason is narrower than "it cannot be done" — because it has been done.

**`polzerStrippingCulturalConductor1990` strips cultural response successfully,
and does it without modeling the grounding at all.** Its central claim is worth
quoting, because it removes the obvious objection:

> we do not have to model the response of a grounded conductor to remove most of
> its effect from the data. This is because most of the vertical magnetic field
> response arises only from the induced currents flowing within the conductor
> itself which need only be estimated by measurement.

The method, at each delay time, decomposes the measurement along a short
traverse across the conductor into two parts — the field of a **unit-ampere
current element** following the conductor's known path, scaled by an unknown
current, plus a **smooth quadratic background** standing for the earth — and
solves least squares for the current. Subtracting the fitted conductor term
leaves the stripped data.

It works, on UTEM data over powerlines and fences, and it is honest about the
residual risk: "it is still necessary to be suspicious of measured responses
close to cultural conductors, particularly if the geometry of the culture is not
well known, or if only a few traverses have been used."

**Three things stop it transferring to production airborne work:**

1. It needs **detailed knowledge of the 3D path of each conductor**. Polzer
   notes skywires are the hard case — "it is impossible to precisely survey the
   location and height of the skywire, particularly as its height must follow a
   catenary curve between suspension points" — and resorts to letting vertices
   float in a steepest-descent fit.
2. It needs **dedicated short traverses across the conductor** to estimate the
   current. A production flight line crosses a powerline once, at whatever angle
   the survey design gives it. There is no traverse to fit.
3. Because the conductor is grounded at many points, the current is not constant
   along it: the conductor "may then be assumed to be composed of a number of
   non-overlapping pieces, each with its own traverse profile", each estimated
   independently. That multiplies the traverses needed.

So stripping is a **survey-design method**, not a processing method. It buys its
result with field effort we do not have in a production AEM delivery.

The modeling route specifically was tried and did not reach ATEM scale.
`blossPowerlineContaminationAirborne2016` built a forward modeling algorithm
around a 'skywire' RL analogue, after Polzer, and reports results "consistent
with the collected gTEM data" but which "do not reproduce the amplitude or
spatial-scale of powerline distortion observed in ATEM data." Parameterising it
properly needs the **contact resistance at each pole** and the **capacitance
between the ground wire and ground**, neither knowable per survey — power
companies target <20 Ω neutral-to-ground (Yin & Wang 2015), but the realised
value varies per pole and with season.

Every crossing is different. Where a pole's grounding is poor there may be
little response at all, so even a calibrated model would not transfer between
two crossings of the same line.

### What Polzer gives us anyway

His decomposition is the ancestor of the profile criterion proposed below.
Polzer fits *smooth background + conductor response* along a traverse in order to
**subtract**; the profile criterion looks for *smooth background + symmetric
departure* along a flight line in order to **detect**.

Detection is the weaker ask, and it is weaker in exactly the places Polzer is
expensive: it needs no conductor geometry, no unit-current template, and no
dedicated traverse — only the observation that the departure is symmetric and
locally confined. Polzer's own result that the stripped response is "symmetric
about the loop with a positive amplitude envelope … of about 40%" is independent
corroboration of the symmetry the 2016 abstract describes.

## Two systems: VTEM and SkyTEM

Both are in scope, and almost every number quoted so far comes from one or the
other rather than from both.

| | SkyTEM | VTEM |
|---|---|---|
| moments | dual — LM and HM | single |
| in this plan | Kang's simulation and field data; Asif's airborne case | `bloss…2016`, `bloss…2017` |
| currently being processed | yes | yes |

**The profile signature is the part that crosses systems.** The double peak with
a central trough was observed in VTEM data in the thesis and derived from 3D
simulation of a SkyTEM geometry by Kang, independently, and they agree. That is
not a small thing: it means the geometric criterion rests on the physics of the
power-line loop rather than on any one system's waveform, and it is the criterion
that should port without retraining.

**The reconstruction criterion is the part that does not.** Three
system-dependencies, in increasing order of how much work they are:

1. **A single-moment system has no LM/HM contrast to exploit.** Kang's finding
   that "the LM response shows a stronger double-peak signature than the HM
   response" is a statement about two moments of one system; on VTEM the whole
   burden falls on one transient, and the strategy of looking at the low moment
   first has no analogue. It also halves the network count — one per system, not
   two.
2. **Retraining is required, and it is a forward-modeling job rather than a
   labeling job.** Asif are explicit that the method "is largely insensitive to
   the system transfer functions, e.g., system waveforms and low-pass filters in
   the receiver system, unless significant changes are made" and "can easily be
   extended for other TEM systems by retraining on forward responses for those
   systems." No labelled VTEM data is needed — which is fortunate, because we do
   not have any.
3. **Raw data stacks may not exist in the delivery.** This is the real obstacle,
   and Asif name it:

   > Obtaining the raw data stacks and the corresponding data uncertainties is
   > not trivial and may not be available for some TEM systems on the normal
   > user/client level. In that case, the thresholds α and β_max may be
   > empirically adjusted to obtain adequate processing results.

   Their uncertainties come from the standard deviation of the raw transients
   within a stacking window, and they add that the window must be "relatively
   short … to obtain good uncertainty estimates and to prevent the coupling
   signatures to be smeared." A delivery that arrives already stacked and
   processed has neither the scatter nor the choice of window.

**What follows for us.** The residual is measured in sigmas, so the denominator
has to come from somewhere. Where a delivery carries raw stacks, use their
scatter as Asif do. Where it does not, the library already has the fallback:
`estimate_noise_model` followed by `assign_uncertainty` puts a modeled
uncertainty on every gate, which is the same denominator `compute_misfit`
already divides by. That is weaker — a modeled sigma cannot see a stack that
happened to be noisy — but it is the difference between the criterion working on
a client delivery and not running at all.

So the ordering in the staging below is not arbitrary. The profile criterion is
second rather than third because it is the one that works on both systems, on
delivered data, with no training artifact.

## How this gets used: two passes

This is a QA/QC tool first and a processing step second, and that changes what
it has to produce.

**Pass one is unattended.** Data arrives, goes through the chain, inverts, and
produces a model and a misfit. Nobody adjudicates individual soundings. The
output is a defensible first answer and the evidence needed to question it.

**Pass two is guided by pass one.** A human reads the first result and adjusts
the culling — more in places, less in others — then reruns. `apply_manual_edits`
(filter_33) is already the checkpoint for exactly this, and `version_diff` and
`parameter_sweep` in the compare group already exist to show what changed.

Three design consequences, all of which cost little if done from the start and
are awkward to retrofit:

- **Write the confidence, not only the verdict.** If the step records `C_avg`
  per sounding as a column alongside the mask, pass two has a continuous field
  to re-threshold and a figure that shows *how close to the line* each sounding
  sat. Without it, "cull more" is a blind re-run; with it, it is a judgement
  about a visible margin. This is the single change that makes the two-pass
  workflow work, and it is one array.
- **The coupling cull must be distinguishable in the provenance.** The library
  already writes `why_` arrays with a step code, so this is nearly free — but
  pass two needs to find *coupling* culls specifically among everything else
  that culled a gate, in order to reverse some of them.
- **Pass two adjusts parameters, not the mask.** Culling is subtractive, so
  nothing can be un-culled in place; a lighter second pass is a rerun from the
  original data with different thresholds, plus manual edits on top. That is
  already how the runner works, and it means the thresholds are the control
  surface — which is what makes the next section matter.

**Default to the lighter hand.** Both failure modes cost, but they differ in
detectability: coupled data left in produces false conductors that a reader can
see and question, while a culled target leaves nothing behind to notice. Pass
one's job is to produce something judgeable, so it should fail in the visible
direction and let pass two tighten. This is the same argument as the 3D
false-positive section above, arriving from the workflow rather than the physics.

A practical note from Asif worth having: inversion of their automatically
processed data "takes fewer iterations to converge as compared to the manually
processed data due to the improved handling of the couplings." If that holds
here, pass one is cheaper than the manual baseline as well as faster.

## Shape of the step

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
  as uncoupled." Loosening it is therefore safer than it looks; tightening it is
  the sharper move.
- **They are not twitchy.** "These parameters are not very sensitive and can be
  adjusted in different ways to obtain similar data processing results", and a
  grid search across several survey areas found one set of thresholds matched
  manual processing best in all of them. The caveat is pointed, though, and it
  is ours: "This scenario may not be true if the signal levels are drastically
  low, e.g., in a very resistive case."

That last point and the raw-stack problem above meet in the same place. Asif's
advice where stacks and uncertainties are unavailable is that "α and β_max may
be empirically adjusted" — which is to say the published defaults are tied to
their uncertainty estimates, and on a delivery where we model the sigmas instead
of measuring them, the two thresholds have to be recalibrated rather than
copied. `parameter_sweep` in the compare group is the tool for that, and the
recalibration is a per-system exercise, not a per-survey one.

**The step's defaults should be Asif's**, because they are the only values with
evidence behind them — with a note in the docstring that they were calibrated
against measured stack scatter on tTEM and SkyTEM, and that a different
uncertainty source moves them.

The powerline monitor **corroborates and never triggers**. No two crossings are
alike, and a pole with poor grounding may produce almost no monitor response, so
a detector gated on the monitor would miss the quiet crossings entirely. Where a
monitor exists and agrees, confidence rises; where it is absent — 3133 soundings
in the one real delivery have a flat monitor — the data decides alone.

## Cost, for a 2700 line-km survey at 30 m

**Inference is free. The million forwards are training, paid once.**

| | |
|---|---|
| soundings | 2700 km ÷ 30 m = **90,000** |
| passes | × 2 moments = **180,000** |
| per pass | ~97k MAC — 24-point input, three 32-filter conv layers |
| total | **≈35 GFLOP**, ~17 MB — seconds on CPU, under a second on GPU |

Asif measured it rather than estimating: **0.3 ms per raw data stack** on one
Xeon Gold 6132 core, fast enough that they propose running it in the instrument
in real time. At that rate the whole survey above is **~54 s on a single CPU
core**, which agrees with the arithmetic and settles the question — inference
cost is not a consideration in this design.

Training: Asif's 1M TEM-resolvable model dataset is **publicly available**, so
the expensive half — forward-modeling von Kármán structures and then *inverting*
them to get resolvable models — may not need regenerating. Their network took
~13,000 epochs at ~6 s/epoch ≈ **22 GPU-hours, once**.

The recurring cost is **one network per system per moment**: SkyTEM LM and HM
are two; HeliTEM, VTEM and Xcite would each need their own. That is a
release-time cost, not a per-survey one, and it is where a GPU forward model
would pay for itself.

## Staging

1. **The framework, with a linear reconstructor.** Residual in sigmas, the
   confidence metric, along-line smoothing, the fuzzy decision, mask and
   provenance. No new dependency. Testable against synthetic couplings now.
2. **The profile criterion** — the double-peak-with-central-trough search. The
   part that addresses the real gap, and the part with nothing to copy.
3. **The auto-encoder reconstructor**, behind the interface from (1), once there
   is a trained artifact and a decision about shipping weights in a
   GPL-licensed repository.

## Open questions

- **Do our deliveries carry raw stacks and measured uncertainties?** This
  decides whether the reconstruction criterion can use Asif's defaults or has to
  be recalibrated against modeled sigmas, and it is answerable today by looking
  at what the real SkyTEM and VTEM deliveries actually contain. Check this
  first — it is cheap and it gates the choice of denominator.
- **If not, what do α and β_max become?** A modeled sigma from
  `estimate_noise_model` has different statistics from measured stack scatter,
  so 2.5 and 0.99 are not transferable on their face. `parameter_sweep` against
  a manually processed line is the way to find out, once per system.
- **Which window** for the profile criterion. Kang's 174 m is for a 10 m × 80 m
  distribution-line loop and does not transfer to transmission lines, whose
  loops enclose an order of magnitude more area. The dependence is known —
  loop area, ground-pathway conductivity, flight height — but the two largest
  unknowns per crossing, contact resistance and catenary sag, both reduce the
  effect and neither is measurable from our data. **Measuring the extent per
  crossing is therefore probably better than deriving it**, with the derived
  value used only to bound the search. Decide which.
- **Does line class need to be an input?** Voltage class proxies for tower
  height and span, and an infrastructure layer usually carries it —
  `cull_by_external_evidence` already accepts such a layer. Using it would
  sharpen the window but makes the criterion dependent on data we often lack.
- **How the profile criterion finds candidates** without being told where the
  powerlines are. With an infrastructure layer it is easy, and
  `cull_by_external_evidence` already accepts one; without, it is an
  unsupervised search for a recurring symmetric pattern.
- **Where a trained model lives.** Weights are not source, the repository is
  going public under GPL-3.0, and a model artifact carries its own provenance
  and licensing questions.
- **Validation.** We have one real delivery and no labelled coupled data, which
  was the hardest gap until Kang. Their notebooks and SimPEG-EM make **synthetic
  coupled data with known ground truth** reachable, and SimPEG is already in the
  stack. That gives a controlled test — vary ground-pathway resistivity, flight
  height and loop count, and measure what the detector catches and misses. It
  does not replace agreement with an expert's manual culling on real data
  (Asif's 96% is the number to beat), but it is a far better starting point than
  nothing, and it is the only way to measure the false-negative rate at all.

## Sources

All keys are in `~/Codes/_refs/library.bib`; PDFs in the YmerFlow shared library.

- `asifAutomatedTransientElectromagnetic2022` — the auto-encoder expert system.
  Section V is the one to reread: the four tunable parameters and what they
  mean, the raw-stack requirement, retraining for other systems, and the
  authors' own statements of the 1D/3D and mineral-exploration limits
- `andersenArtificialNeuralNetworks2016` — the earlier supervised ANN
- `larsenRemovalPowerlineNoise2022` — harmonic removal; **not applicable here**
- `xieImprovedPrincipalComponent2014` — PCA denoising; a reconstruction basis
- `blossPowerlineContaminationTimedomain2017` — the signatures and the profile
- `blossPowerlineContaminationAirborne2016` — the quoted characterisation
- `macnaeQuantifyingAirborneInduced2016`, `macnaeQuantitativeEstimationIntrinsic2016`
  — airborne IP quantified and estimated: the oscillatory signature is
  recoverable signal, not only a nuisance
- `grombacherInducedPolarizationEffects2021` — IP effects in real ATEM data
- `kangEffectPowerLines2021` — 3D simulation of the mechanism; the
  flux-cancellation explanation of the trough, the distance numbers, and
  released code that makes synthetic coupled data reachable
- `polzerStrippingCulturalConductor1990` — stripping by measured current;
  the ancestor of the profile decomposition
- Yin & Wang 2015, powerline ground resistance from helicopter EM. **Not read**
