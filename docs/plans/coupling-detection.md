# Detecting coupled data

**Status:** planned, not started. Literature reviewed 2026-10-15.
**Decided:** detect and cull only, never repair. The powerline monitor
corroborates; it does not trigger.

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

`blossPowerlineContaminationTimedomain2017` Fig. 1-1 separates them:

| | what it looks like |
|---|---|
| **Capacitive** | oscillatory behavior at later times |
| **Inductive** | **smoothly elevated** dB/dt curves |

The existing culling filters — `cull_negatives`, `cull_by_curvature`,
`cull_by_decay_rate` — are shape tests on one sounding at a time. They catch the
oscillatory kind, because oscillation breaks monotonicity and curvature.

**They cannot catch smooth elevation.** An inductively coupled sounding is still
a plausible-looking decay; it is simply in the wrong place. Nothing about its
shape is wrong.

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

A per-sounding detector run across this pattern does the wrong thing twice: it
flags the flanks and **keeps the centre**, leaving an island of apparently good
data directly over the conductor.

Detecting the pattern rather than the sounding is the contribution available
here, and it is ours.

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

## Shape of the step

One filter, several criteria, following `cull_by_external_evidence` — which
already takes a `criterion` and has the `proximity`/`misfit`/`both` precedent.

```yaml
- cull_by_coupling:
    criterion: reconstruction     # reconstruction | profile | both
    max_residual: 2.5             # sigmas; Asif's alpha
    confidence: 0.99              # Asif's beta_max
    window: 7                     # soundings smoothed over; Asif's W
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

- **Which window** for the profile criterion, given 300 m and ~1 km both appear
  and the extent is time-gate dependent.
- **How the profile criterion finds candidates** without being told where the
  powerlines are. With an infrastructure layer it is easy, and
  `cull_by_external_evidence` already accepts one; without, it is an
  unsupervised search for a recurring symmetric pattern.
- **Where a trained model lives.** Weights are not source, the repository is
  going public under GPL-3.0, and a model artifact carries its own provenance
  and licensing questions.
- **Validation.** We have one real delivery and no labelled coupled data. The
  honest acceptance test is agreement with an expert's manual culling where that
  exists; Asif's 96% is the number to beat, and we cannot measure ours without
  labels.

## Sources

All keys are in `~/Codes/_refs/library.bib`; PDFs in the YmerFlow shared library.

- `asifAutomatedTransientElectromagnetic2022` — the auto-encoder expert system
- `andersenArtificialNeuralNetworks2016` — the earlier supervised ANN
- `larsenRemovalPowerlineNoise2022` — harmonic removal; **not applicable here**
- `xieImprovedPrincipalComponent2014` — PCA denoising; a reconstruction basis
- `blossPowerlineContaminationTimedomain2017` — the signatures and the profile
- `blossPowerlineContaminationAirborne2016` — the quoted characterisation
- `polzerStrippingCulturalConductor1990` — stripping by measured current;
  the ancestor of the profile decomposition
- Yin & Wang 2015, powerline ground resistance from helicopter EM. **Not read**
