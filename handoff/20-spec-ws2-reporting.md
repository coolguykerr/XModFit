# Spec: WS2, the joint-fit reporting contract, on top of general covariance

**Written 2026-11-11. Status: approved by the user, 2026-11-11; amended the same day to
use the exact likelihood in `model` mode (§3.5, §4.1).** The task-by-task plan is
`21-plan-ws2-reporting.md`, as doc 18 followed doc 17.

`14-roadmap.md` §WS2 already fixes most of this workstream: the module, the functions, the
headline number and the done-when tests. This document does three things the roadmap could
not:

1. It settles **how χ² splits by dataset** now that every fit can carry a covariance,
   including covariance terms shared between datasets (spec 17 §10 left this to WS2).
2. It makes every weighted objective an **exact Gaussian log-likelihood**, including
   `model` mode, where the covariance depends on the fitted model (§3.5, amended by the
   user 2026-11-11). That replaces the earlier device of re-scoring two fits under one
   covariance (§4.1).
3. It settles roadmap Decisions 2, 3 and 5, and revisits two roadmap rules that the Log fix
   (`64babd9`) and the absolute-error decision (spec 17 §6.2) overtook.

Everything the roadmap says about WS2 stands unless a section here says otherwise.

## 1. Intent

A joint fit currently gives the user a lmfit report and, separately, a comparison panel
that headlines AIC/BIC computed on a plain χ² that the fit never minimised. The user ends
up reading numbers that are either the wrong objective (plain χ² under a GLS fit) or not a
confidence measure (a bare Δχ² between two fits of 17 datasets). WS2 makes the program
compute the numbers worth reading, from **the fit's exact objective** (the one it
minimised, or in `model` mode the exact likelihood at the fitted point, §3.5), and
say what each number was computed from.

## 2. Goals, non-goals, anti-goals

**Goals**
- A per-dataset split of the fit's objective that is exact, for every fit scale and every
  mix of datasets with and without a covariance (§3): it sums to `result.chisqr`, or in
  `model` mode to the objective at the fitted model (§3.5).
- For every weighted objective, the exact −2 ln L, log-determinant included, split by
  dataset by the chain rule (§3.5).
- A comparability guard and a calibrated comparison between two fits of the same data
  (§4, §5), usable from scripts now and by WS3's library later.
- Per-dataset quality against the individual fits, and the end-of-fit diagnostics of
  roadmap §2.5 (§6), printed in the joint fit report.
- Leave-one-out refits as a function (§7).
- Opt-in per-dataset error rescaling by √χ²_red (§6.3), off by default.

**Non-goals**
- Any GUI change. The comparison panel (`computeModelComparison`) keeps its current text
  until WS4 swaps in `format_joint_vs_individuals` (roadmap §2.7).
- Storage of fits across sessions (WS3). WS2 defines the in-memory snapshot WS3 will
  persist (§3.1).
- Single-dataset `Fit` diagnostics. Their χ²_red band already exists (doc 17). The
  functions here take a `JointFit`; a one-dataset joint fit works through the small-N
  rule.
- An analytic joint Jacobian (WS5). WS2 uses finite differences where it needs a Jacobian.
- Changing how a `model`-mode fit or MCMC run **optimises or samples**. WS2 reports the
  exact likelihood at the fitted point and how far the fitted point is from that
  likelihood's optimum (§6.4 item 6). Sampling and optimising the exact likelihood is
  roadmap WS6 (§3.5 says why it is cheap).

**Anti-goals** (failure even if it technically works)
- A number shown without the objective it came from.
- A calibrated z printed from 2–3 datasets.
- Two fits compared when they scored different points, data or covariances.
- A diagnostic failure that breaks or alters a fit. Diagnostics follow the error policy:
  on failure the block says what could not be computed and the fit is untouched.

## 3. The per-dataset split (roadmap §2.1)

### 3.1 The snapshot

`snapshot(jf) -> JointSnapshot` runs once at the end of a fit, at the fitted parameters,
with **no refit**. It evaluates the joint residual once, through the same code path the fit
used, and records, per dataset in fit order:

- name, active index (and its hash), window (xmin, xmax), fit scale;
- `x`, `y`, `yerr` on the active rows (the key hashes them; a dataset without a
  covariance has `yerr` as its only σ);
- the row range this dataset occupies in the residual vector actually minimised. This is
  not stored by `JointFit` today: it follows from the per-dataset active counts and
  `_cov_mask` (`_build_chol`), and `JointFit` exposes it;
- the residual `r_i` in the fit's scale (d − f in Linear, ln d − ln f in Log) and the
  model `f_i`, on the active rows;
- two intensities for this dataset. **`fit_intensity`** is the one the fit was scored
  under (`Block.intensity` in the last `_build_chol`). In `model` mode that is the
  *previous* round's model, not `f_i`, because the final round of `outer_iterate` fits
  under the C rebuilt from the round before. **`intensity`** is the one at the fitted
  model, built from `f_i` by the same rule `_build_chol(model=True)` uses (the data
  substituted on active rows where the model is non-finite, or ≤ 0 in Log). The two are
  identical except in `model` mode. Everything in §3–§5 uses `intensity`, the exact
  likelihood at θ̂ (§3.5). `fit_intensity` exists to reproduce `result.chisqr` and
  report the gap;
- the covariance object and mode **as used**: `None` when the covariance was ignored
  (a `w/o error` scale, a fit-time mismatch), with the downgrade note.

And, for the fit as a whole: the whitened residual `z` under C(θ̂), the whitened
residual the fit actually minimised (identical to `z` except in `model` mode), the
covariance mask, the jitter added by `factorize` (if any), the prior rows (`set_priors`
appends them after whitening), the method, `nvarys`, `ndata`, parameter values, stderrs,
`covar`, `var_names`, the `param_map`, and the fit warnings.

It does **not** store C or L (N² numbers). They are rebuilt on demand from the stored
covariance objects, intensities, `y`, `yerr` and mask through `covariance_core.assemble`,
plus the stored jitter on the diagonal. **Everything in §3–§5 reads only the snapshot, never
a model.** §6 and §7 evaluate models, because they run while the live `JointFit` exists.

**No side effects.** `JointFit.residual` increments `Niter`, emits `functionCalled` (a Qt
signal the GUI plots from) and `evaluate()` overwrites each model's `x` and fit arrays. The
snapshot, the Fisher Jacobian (§6.2) and LOO (§7) must evaluate without emitting, without
changing `Niter`, and leave every model as it was. That needs a quiet evaluation path in
`JointFit`, or a save-and-restore around the existing one.

**Which fits.** The identity with `result.chisqr` holds for a completed least-squares fit,
using `fit_intensity`. Two cases get a snapshot whose objective is recomputed and
labelled, and no identity is claimed:
- an MCMC result: the snapshot is taken at the reported parameter values (lmfit's
  posterior medians), labelled "at posterior medians". A `model`-mode MCMC run sampled
  with C frozen at the prior fit (`perform_fit` freezes it, `Joint_Fit_Routines.py`
  ~1307–1321). The label adds "sampled with C frozen at the prior fit, not the exact
  likelihood";
- a model-mode fit the user aborted: `rebuild` has already replaced `L` when the old
  result is returned, so the recomputed χ² is labelled "aborted fit; not the minimised
  value".

### 3.2 The split

The fit minimises `‖z‖²` with `z = L⁻¹ r`, where `L` is the Cholesky factor of the full
covariance over the concatenated active points, in fit order (`JointFit._build_chol`).
Because `L` is lower triangular, the rows of `z` belonging to dataset i depend only on
datasets 1..i. So:

- **Sequential term** `s_i = ‖z_i‖²`, the dataset's slice of z with L the Cholesky factor
  of C(θ̂) (§3.1). `Σ s_i + (prior rows)²` is the objective at the fitted model. It equals
  `result.chisqr` exactly except in `model` mode, where the fit's own value comes from the
  minimised residual and the difference is reported (§3.5). Each `s_i ≥ 0`. Statistically
  it is the χ² of dataset i *conditional on* datasets 1..i−1, and under the model the
  `z_i` are independent standard normals, so the `s_i` are independent χ²(n_i) variables.
  This is the chain-rule factorisation of the joint likelihood,
  p(y) = Π p(y_i | y_<i).
- **Marginal term** `m_i = r_iᵀ C_ii⁻¹ r_i`, dataset i scored under its own diagonal
  block of C (with any `factorize` jitter included, i.e. the block of `L Lᵀ`, so that
  `m_i = s_i` holds exactly when C is block-diagonal), i.e. as if fitted alone with these parameter values. It is what an
  individual fit of dataset i would report at these parameters. It is order-independent.
- **Coupling** `= Σ_i s_i − Σ_i m_i`: the part of the joint objective that exists only
  because datasets share error terms. Zero when C is block-diagonal.

**When C is block-diagonal**, which is every fit without a shared term, `s_i = m_i` and the
split is unique and exact. This includes every no-covariance fit (where `s_i` is the sum of
the dataset's squared weighted residuals, as the roadmap specified), every fit with
LSS_Reader sidecars (their terms are `scope: "dataset"`), and mixed fits. **The two only
differ with shared terms**, which today means the legacy bilayer bundle
(`attenuator_cal`, `beam_fwhm`).

**Which term is used where [call; this replaces roadmap Decision 2]:**

| use | term | why |
|---|---|---|
| sums, Δ(−logL), the calibrated comparison (§5) | sequential `ℓ_i` (§3.5) | sums exactly; terms independent under the model, which is what the `sd(d)·√N` scaling of z assumes |
| per-dataset quality χ²/n | marginal `m_i` | comparable with the dataset's own individual fit, which only ever sees `C_ii` |
| joint vs individual (§6.1) | marginal `ℓm_i` (§3.5) | as above, and a likelihood when C depends on the model |
| Fisher shares (§6.2) | marginal block `C_ii` | as above |

The cost of the sequential split is that it depends on dataset order: the first dataset
carries its marginal χ², later ones are conditional. The order is part of the
comparability key (§4), so two compared fits always split the same way, and the report
prints the order when a coupling is present.

The roadmap's recommendation was the additive Mahalanobis split `t_i = r_iᵀ(C⁻¹r)_i`. It is
order-independent and sums exactly, but a term can be negative and the terms are correlated,
so neither "χ² of dataset i" nor an `sd`-based z means anything for it. It is not used.

### 3.3 `DatasetTerm`

```
DatasetTerm: name, n_points, sequential, marginal, chi2_per_point (= marginal / n),
             logdet_seq, logdet_marg, nll2_seq, nll2_marg (§3.5; None when unweighted),
             window (xmin, xmax), index_hash, scale, cov (bool), cov_mode (str|None)
per_dataset_objective(snap) -> (list[DatasetTerm], extras)
extras: prior (float), coupling (float), order (list of names), objective_label (str),
        nll2 (float | None: Σ nll2_seq, the data's −2 ln L), fit_gap (float | None, §3.5)
```

Per-dataset **χ²/n, not χ²_red** [call]. With shared parameters there is no unique way to
assign degrees of freedom to datasets, and the user has rejected χ²_red as a variance
estimator for these models (spec 17 §6.2). The fit-level χ²_red (with N − p) stays where it
is. The roadmap's `rescale` field becomes the factor `s_i` of the opt-in rescaling (§6.3),
present only when that option is on.

### 3.4 What the objective is called (roadmap global constraint, revised [call])

Every number carries one of these labels:

| fit scale | covariance | label | likelihood? |
|---|---|---|---|
| Linear | none | χ² (Linear, weighted by the error column) | Gaussian: −2 ln L = χ² + log det + n ln 2π |
| Linear | attached | GLS χ² (Linear, covariance `<mode>`) | Gaussian, as above (§3.5) |
| Log | none | χ² of ln y (Log, σ_ln = σ/y) | Gaussian in ln y, to first order |
| Log | attached | GLS χ² of ln y (Log, covariance `<mode>`) | Gaussian in ln y, to first order |
| Log w/o error | — | unweighted sum of squares of log₁₀ y | **no** |
| Linear w/o error | — | unweighted sum of squares of y / max(y), per dataset | **no** |

**`data` mode carries one more sentence** in its label: "covariance built from the
measured data; approximate". With multiplicative terms, the data-based C weights each
point by its measured, not its expected, intensity. That is the setting of the known
low bias in fits with normalisation errors (Peelle's Pertinent Puzzle). It is still a
fixed Gaussian, so its −2 ln L is computed and compared the same way; the sentence says
what it approximates. `model` mode carries no such sentence.

**This revises a roadmap rule.** The roadmap forbade calling any `Log` number a likelihood,
because the old Log residual used log₁₀ and understated χ² by ln 10. Since `64babd9` the
weighted Log residual is (ln y − ln f)·y/σ, a first-order Gaussian likelihood for ln y, and
the covariance Log path uses C_log of ln y. A Δ(−logL) between two Log fits is therefore
legitimate, labelled "of ln y, first order". A Log fit is never compared with a Linear fit:
they are likelihoods of different variables, and the key refuses it (§4).

### 3.5 The exact likelihood, including `model` mode (amended by the user, 2026-11-11)

**What C(θ) is.** `covariance_core.assemble` builds
C(θ) = Σ_meas + Σ_t G_t(θ) Σ_t G_t(θ)ᵀ. Σ_meas is the fixed diagonal measurement
variance, and Σ_t is a term's covariance. G_t = diag(D)·W_t for a multiplicative term in
Linear, where D is the intensity: the data in `data` mode, the model f(θ) in `model` mode.
This is exactly the covariance of y = f(θ)∘(1 + W a) + ε with a ~ N(0, Σ_t) and ε ~ N(0,
Σ_meas), once the factors a are integrated out. A `model`-mode C is therefore not a
reweighting heuristic. Everything that defines the error comes from the reduction:
Σ_meas, the terms' Σ_t and W, from LSS_Reader or the attached file. The model supplies
only D, the intensity that turns a **relative** scale-factor error into an absolute one.
A scale error ε on scan k moves each of its points by ε·(true intensity), and the true
intensity is unknown: `data` mode estimates it by the measured y, `model` mode by f(θ).
The data estimate correlates the weights with the noise (a point that fluctuated low is
given a smaller error and more weight), which is the source of the data-mode bias in
§3.4. In Log the multiplicative terms do not involve D at all, and the two modes agree.
With D = f(θ), C(θ) is the covariance of one fixed generative model. The likelihood
−2 ln L(θ) = r(θ)ᵀ C(θ)⁻¹ r(θ) + ln det C(θ) + N ln 2π
is one fixed function of θ, a single surface. Nothing shifts: what changes from point to
point is the prediction, as the mean does. This holds exactly for multiplicative terms in
both scales, and to first order for terms whose loading is itself a linearisation
(additive terms in Log, and any term built from a derivative).

**What the report computes from it.** In L = chol(C(θ̂)), the sequential split extends to
the log-determinant row by row, because ln det C = 2 Σ_j ln L_jj:

- `logdet_seq_i = 2 Σ_{j ∈ rows_i} ln L_jj` and
  `nll2_seq_i = s_i + logdet_seq_i + n_i ln 2π`. This is exactly
  −2 ln p(y_i | y_1..y_{i−1}), the chain rule of §3.2 applied to the likelihood itself,
  and `Σ nll2_seq_i = −2 ln L` of the data (priors separate).
- `logdet_marg_i = ln det (L Lᵀ)_ii` and `nll2_marg_i = m_i + logdet_marg_i + n_i ln 2π`,
  which is −2 ln p(y_i) on its own.
- Without a covariance, `logdet` is Σ ln σ_i² over the rows (σ the weight the residual
  used: `yerr` in Linear, `yerr/y` in Log, times `s_i` when rescaled, §6.3). Unweighted
  scales have no likelihood, so these fields are `None`.

In `data` mode and without a covariance, the log-determinant is the same for every fit
the key admits (§4), so it cancels in every difference. It is reported for completeness.
In `model` mode it depends on θ and is part of every comparison.

**The fit gap.** The reported point θ̂ is `outer_iterate`'s endpoint. Its final round
minimised r(θ)ᵀ C(θ_prev)⁻¹ r(θ). That is not the exact objective, for two reasons:
C(θ_prev) ≠ C(θ̂), and the fixed point ignores the gradient through C(θ). The snapshot
reports `fit_gap = (exact χ² at θ̂) − result.chisqr`, and §6.4 item 6 reports how far
θ̂ is from the exact likelihood's optimum. Both are zero, and not printed, outside
`model` mode with intensity-dependent terms.

**Why sampling and optimising it is cheap, for WS6** (not built here). Without dense
terms, C = Σ_meas + G Σ_a Gᵀ is a diagonal plus a rank-K term, where K is the number of
scale-factor components, tens against N of thousands. Woodbury and the matrix
determinant lemma give χ² and ln det C(θ) in O(N K²) at every θ. That is no more than
the O(N²) triangular solve with a frozen dense L that MCMC does today, while K ≲ √N.
Sampling with C(θ) recomputed at every step therefore costs about what the current
frozen-C run costs, and it targets the exact posterior. Integrating a out analytically
in this way gives the same posterior for θ as sampling the explicit factors a with
fixed diagonal weights, with K fewer dimensions. The ln det term must then be in the log-probability, or the
sampler is rewarded for inflating f to inflate the variance. Dense terms in Log are the
one case that needs a full factorisation per step.

## 4. Comparability (roadmap §2.2)

```
comparability_key(snap) -> dict
```

Two fits are compared only if their keys agree on:

- the datasets, in **order** (order fixes the sequential split, §3.2), each by name and a
  hash of (x, y, yerr) over its active points;
- each dataset's window and active-index hash;
- the fit scale;
- each dataset's covariance **as used**: none (including one attached but ignored), or
  its source, a content hash and the mode;
- the error-rescaling option (§6.3): off, or `'per_dataset'` with its `inflate_only` flag.

A refusal names the first differing field, e.g. "dataset 3 window 0.030:0.450 vs
0.030:0.500".

**Allowed to differ [call; revises the roadmap, which put model class and PDB in the
key]:** parameter state (free, fixed, shared, bounds), the model class, the PDB template,
the priors and the fit method. Comparing two different models on the same data is the
non-nested case the calibrated comparison is designed for. The comparison reports the
differences ("different model: XLayers vs XLayersPDB") and uses them to decide nesting
(§5). Priors are not part of the data terms. A prior difference is reported on its own line
and never enters Σd.

### 4.1 Model-mode fits: each under its own C(θ̂), log det included (amended 2026-11-11)

In `model` mode, C depends on the fitted model: D = diag(f(θ̂)) for multiplicative terms
in Linear, and the intensity of additive and dense terms in Log. Two fits A and B of the
same data with the same covariance file are scored under C(θ̂_A) and C(θ̂_B). Their χ²
alone are not comparable, because the weights moved between the fits.

**Rule:** each fit is scored under its own C(θ̂) (§3.1, `intensity`), and the comparison
uses the full `nll2` terms of §3.5, log-determinant included. The key guarantees that
both fits use the same Σ_meas, the same terms with the same Σ_t and W, and the same mode.
They are therefore two points on **one** likelihood surface (§3.5), and Δ(−2 ln L) is an
exact likelihood ratio. No re-scoring under the other fit's covariance is done, and no
model is evaluated. The report's detail line shows Δχ² and Δ log det separately, so the
reader sees how much of the difference came from the covariance.

An earlier draft re-scored both fits under the reference fit's covariance. That gave a
weighted least-squares comparison at fixed weights, not a likelihood ratio, and it could
disagree with itself depending on which fit was the reference. It is superseded.

## 5. The calibrated comparison (roadmap §2.3)

```
compare_fits(a: JointSnapshot, b: JointSnapshot) -> Comparison
Comparison: d (per dataset), sum_d, sd_d, N, z, sign_count, delta_k, nested,
            worst_dataset, objective_label, delta_chi2, delta_logdet, notes
```

As the roadmap specifies, with these settled:

- `d_i = ½(nll2_seq_i^A − nll2_seq_i^B)` for a likelihood objective (§3.4, §3.5), so
  Σd = Δ(−ln L) over the data, exactly. For an unweighted objective,
  `d_i = s_i^A − s_i^B`, labelled "difference in sum of squares; not a likelihood".
- `delta_chi2` and `delta_logdet` split Σd into its two parts. `delta_logdet` is 0
  except in `model` mode, and with rescaling (§6.3).
- **Headline:** `Δ = Σd ± sd(d)·√N (z = |Σd| / (sd(d)·√N))`, which fit is preferred, and
  the sign count ("B preferred on 14/17 datasets").
- Under a coupling (§3.2) the z line adds: "datasets treated as independent after
  conditioning on shared error terms in the listed order".
- **Small-N rule, unchanged:** N < 4 datasets shows the `d_i` table and the sentence "no
  calibrated z: too few datasets to estimate the spread". No z is printed.
- **Nesting:** B is nested in A if every master of B is a union of masters of A (sharing
  coarsens A's partition of dataset-parameter endpoints), every endpoint free in B is free
  in A, and the model class and PDB agree. Then Δk is shown, and the likelihood-ratio
  p-value is shown with the caveat that the model is likely misspecified and that LOO (§7)
  is the honest check.
- The bare Δ(−logL) appears only in the detail line, labelled "not a confidence measure",
  with its χ² and log-det parts.

`format_comparison(c) -> str` renders it.

## 6. End-of-fit diagnostics (roadmap §2.4, §2.5)

`diagnostics(jf, snap, individual_fits=None) -> dict`, rendered by `format_diagnostics`,
is run at the end of every joint fit and appended to `JointFit._report()` under
"Joint-fit diagnostics". It is wrapped so that any failure produces a line
"diagnostics unavailable: <reason>" and leaves the fit and the rest of the report
untouched.

### 6.1 Per-dataset quality, and joint vs individual fits

One row per dataset: n, χ²/n (marginal), and, when an individual fit of that dataset is
supplied, its value **re-scored on the joint fit's active points** (the individual model
is evaluated once on those points, quietly, §3.1). Then the cost of sharing,
`Δ_i = nll2_marg_i^joint − nll2_marg_i^individual`, in −2 ln L units. It equals
`m_i^joint − m_i^individual` whenever C_ii does not depend on the model.

- `individual_fits` is a dict `{dataset name: Fit}`, as in the GUI's `dataset_fits`. The
  `Fit` supplies its model and parameters; the covariance and mode are the joint fit's for
  that dataset, so both sides use the same covariance file.
- Each side is scored under its own C_ii, by the §4.1 rule: the joint side under the
  snapshot's block at θ̂_joint; the individual side, in `model` mode, under a C_ii built
  from the individual model's intensity by the same substitution rule (§3.1). Both are
  points on one likelihood for dataset i, so Δ_i needs no cross-check.
- A row is "unavailable" with the reason when the individual fit used a different fit
  scale, or has no result.

Rows are flagged where Δ_i is large against the others: `Δ_i > median(Δ) + 3·1.4826·MAD(Δ)`
and `Δ_i > 2` [call]. This is roadmap hazard 4, "a shared value that fits nothing well".

`JointFit._report()` has no access to the GUI's individual fits, so the report shows the
columns it can. `format_joint_vs_individuals(jf, individual_fits)` produces the full table
and the joint-vs-individuals calibrated comparison (§5 machinery over `m_i`) for scripts
now and for the WS4 panel swap.

### 6.2 What each dataset tells you about the shared parameters

The Fisher information for each shared master parameter, per dataset, from a
finite-difference Jacobian of the residual at the fitted values (one model evaluation per
free master, forward difference, step chosen by the builder and tested against an
analytic case):

- per dataset, **marginal**: `I_i = J_iᵀ C_ii⁻¹ J_i` (the information dataset i carries on
  its own);
- the table shows each dataset's share `I_i[p,p] / Σ_j I_j[p,p]` for each shared p.

Under a coupling the shares are of the marginal informations, and the report says the joint
information is smaller than their sum because the shared errors are counted once.

**Omitted, and said so [call]:** in `model` mode the Gaussian Fisher information has a
second term, ½ tr(C⁻¹ ∂_pC C⁻¹ ∂_qC), the information the covariance itself carries
about θ. It is omitted. ∂C/∂θ has rank at most 2K, so the term grows with K, not with
N as the mean term does. Each of its at most 2K contributions also shrinks with the share
of the variance that the scale factors carry. The table's footnote says "mean information only". The same omission is in
lmfit's stderrs today, which come from the whitened Jacobian at fixed C.

### 6.3 Error-bar heterogeneity: reported always, rescaling opt-in (decided by the user, 2026-11-11)

**Always:** report the spread of per-dataset χ²/n, and when max/min exceeds 3
(user-settable), say so: "error bars or model adequacy differ between datasets".

**Opt-in: per-dataset error rescaling (Birge-style).** The user wants it available because
it is the conventional remedy and XModFit serves a broad range of users. It is **off by
default** and never switched on by the software.

- **Switch:** `JointFit.rescale_errors` ∈ {`None` (default), `'per_dataset'`}. Set in code
  or in the `.jpar`; the GUI control is WS4.
- **Factor:** `s_i = √(χ²_i / ν_i)` with χ²_i the **marginal** term (§3.2) and
  `ν_i = n_i − p·n_i/N` (the fit's p free parameters apportioned by points, so
  `Σν_i = N − p`) [call: there is no unique per-dataset dof with shared parameters; this
  one is documented and printed]. Both directions, as is conventional (s_i < 1 shrinks
  errors); an `inflate_only=True` variant uses `max(1, s_i)`.
- **Application:** the dataset's error column is multiplied by s_i, or with a covariance
  the whole covariance is transformed `C → S C S` with `S = diag(s_i)` on each dataset's
  rows (so a shared term's cross block scales by s_i·s_j and C stays PSD).
- **Procedure:** fit; compute s; refit warm-started under the rescaled errors; repeat until
  every s_i changes by < 1 % or 5 rounds (report which). The final s_i are stored in the
  snapshot and the `.jpar` result.
- **Labelling:** every number from a rescaled fit carries "errors rescaled per dataset by
  √χ²_red (s = …)". Its objective is **not** a known-error likelihood: §3.4 gives it the
  label "rescaled χ²", and §5 refuses to compute z or a likelihood-ratio p-value between a
  rescaled and an unrescaled fit.
- **What its likelihood is (amended 2026-11-11):** with S in the covariance,
  ln det(S C S) = ln det C + 2 Σ_i n_i ln s_i, and that term is in its `nll2` (§3.5). The
  rescaled fit's −2 ln L is then the likelihood with one unknown error scale per dataset,
  evaluated at the fitted s. It is approximate, because s uses ν_i where the maximum
  likelihood value would use n_i. The label says "error scales fitted per dataset;
  approximate", and §8 counts the N scales in k.
- **Comparability:** `rescale_errors` (mode and `inflate_only`) is in the key (§4). Two
  rescaled fits are compared each under its own s, through `nll2`, as in the §4.1 rule.
  There is no re-scoring.
- **Why it is opt-in and not the default:** a large s_i is usually misfit, not
  under-estimated errors. Rescaling makes that dataset's misfit look acceptable and
  down-weights it in the shared parameters, so the report always prints the unrescaled
  per-dataset χ²/n next to the factors.

### 6.4 The rest of roadmap §2.5

1. **q-range spread:** `max(qmax)/min(qmax) − 1 > 10 %` (user-settable) is reported with
   the window mode in use ("native windows").
2. **Shared-parameter consistency:** the WS1 first-endpoint warnings, restated in the
   block.
3. **Degeneracy:** correlation pairs with |ρ| > 0.9 from `result.covar`, and the softest
   direction, the eigenvector of the correlation matrix with the largest eigenvalue,
   written in parameter terms with coefficients in units of each parameter's stderr
   ("d_HGi + 0.8·d_wat"). This reads only `result.covar`, so it costs no model
   evaluation. When `covar` is missing (a failed Hessian), say so.
4. **What was minimised:** the §3.4 label, the covariance mode, and the dataset order when
   there is a coupling.
5. **Model–reduction tension (spec 17 §4.6), optional:** for each multiplicative component
   of each covariance, the shift the fit leaned on, `ε̂ = Σ Wᵀ D C⁻¹ r`, and its pull
   `ε̂_k / √Σ_kk`. Labelled *model–reduction tension*, never *stitching quality*, and never
   presented as a fitted scale factor.
6. **Distance to the exact optimum (`model` mode only, amended 2026-11-11):** the gradient
   g of the exact objective Φ(θ) = −2 ln L(θ) + (prior rows)² at θ̂, by central
   differences. Each evaluation is one quiet model evaluation plus one assemble and one
   factorisation. The Newton step to Φ's optimum is δ = −½·covar·g (Φ's Hessian is about
   2·covar⁻¹, since covar = (JᵀC⁻¹J)⁻¹ with `scale_covar=False`). The report prints
   `max_p |δ_p| / stderr_p` and the parameter attaining it, as "the fitted point is the
   reweighting fixed point; the exact-likelihood optimum is estimated δ away".
   `fit_gap` (§3.5) is printed on the same line. Skipped, with no line, when C does not
   depend on the model, and "unavailable" when `covar` is missing. When δ is not small
   (> 0.3 stderr), the line adds that the WS6 exact optimiser should settle it.

## 7. Leave-one-out (roadmap §2.6)

`leave_one_out(jf, workers=1) -> LOOResult` refits N times, each omitting one dataset and
warm-started from the full fit, on copies, never mutating `jf`. Covariance rows of the
omitted dataset are dropped, which is exact marginalisation, so shared terms stay correct.
It reports, for each shared master parameter, the spread of the N estimates and each
dataset's influence (the shift when it is omitted, in units of the full-fit stderr).
`format_loo` renders it. It is a function only, not run at the end of a fit (cost: N full
fits).

**Deferred to a later plan [call]:** anchor-and-verify and the sign-agreement pre-check
(roadmap §2.6). Both need decisions about subset choice that are easier once LOO output
exists on real data.

## 8. AIC and BIC (roadmap Decision 3)

Kept, demoted below the calibrated comparison under "information criteria (assume a
correct model; read with caution)" [call]. Corrected to the known-error form: with
weighted errors, `AIC = χ² + 2k` and `BIC = χ² + k ln N`, not the `N ln(RSS/N)` form in
`computeModelComparison` (`xmodfit.py:8053`), which assumes the error scale is unknown and
fitted. They are computed on the §3 objective, as −2 ln L (§3.5, `extras['nll2']`) + 2k and
+ k ln N. This equals the χ² form up to a constant shared by every fit the key admits,
except in `model` mode, where the log-det is part of it. A rescaled fit counts its N
error scales in k (§6.3). For an unweighted objective the RSS form is the right one and is
used. The GUI panel keeps its old text until WS4.

## 9. Where the code goes

- New Qt-free root module `joint_fit_stats.py`, next to `covariance_core.py`. Not in a
  `Functions/<category>/` folder (CLAUDE.md rule 1).
- `JointFit._report()` gains the diagnostics block. That is the only change to an
  existing file outside tests, apart from `JointFit` exposing what `snapshot` needs
  (row ranges, the prior rows), if it does not already, and the opt-in rescaling of §6.3:
  `rescale_errors` on `JointFit`, the refit loop in `perform_fit` (off by default, so the
  default path is untouched), and its `.jpar` field.
- `_eval_chisqr` and `computeModelComparison` are **not** touched (WS4). A1.4's "superseded
  by WS2 §2.1" is satisfied by the new functions existing and being used by the report.

## 10. Testing

All headless, in `tests/run_all.py`, each shown to fail when its feature is reverted.

1. **No covariance:** sequential = marginal = the hand-computed per-dataset sums, and
   `Σ + prior = result.chisqr` to 1e-12, in Linear and Log.
2. **Block-diagonal covariance** (private terms, a mixed fit): sequential = marginal,
   coupling = 0, sum = `result.chisqr` to 1e-10.
3. **Shared term:** sum of sequential = `result.chisqr` to 1e-10; the first dataset's
   sequential term equals its marginal; reversing the dataset order leaves the sum
   unchanged and changes the terms; marginal computed independently from C_ii; coupling =
   sum − Σ marginal.
4. **Exact likelihood (§3.5):**
   - Σ logdet_seq = `np.linalg.slogdet(C)` and Σ nll2_seq equals an independent dense
     Gaussian log-pdf (`scipy.stats.multivariate_normal.logpdf`) to 1e-10, in Linear and
     Log, with a shared term.
   - In `model` mode the snapshot's C equals one assembled from f(θ̂).
   - `fit_gap` is reported, and the `fit_intensity` path reproduces `result.chisqr`.
   - On a constructed Linear `model`-mode pair, Σd differs from ½Δχ² by exactly
     ½Δ log det. Two Log multiplicative-only fits have Δ log det = 0.
   - The distance-to-optimum δ (§6.4 item 6) agrees in sign and to 20 % with a direct
     `scipy.optimize.minimize` of Φ on a small synthetic problem.
5. **compare_fits:** two synthetic snapshots with a known `d` reproduce Σd, sd, z and the
   sign count exactly; N = 3 prints no z; each key field, changed alone, is refused with
   its name; a different model class is compared and flagged non-nested; a nested pair
   gets Δk.
6. **Labels:** every row of §3.4, including "not a likelihood" for both `w/o error`
   scales.
7. **Fisher:** a linear synthetic joint model reproduces the analytic `JᵀC⁻¹J` per
   dataset to finite-difference precision; shares sum to 1 when block-diagonal.
8. **Degeneracy:** a constructed exactly-degenerate pair is named as the softest
   direction with the right coefficient sign.
9. **Diagnostics failure:** a diagnostic forced to raise leaves the fit result and the
   rest of the report unchanged and prints the "unavailable" line.
10. **LOO:** on a linear synthetic joint fit it matches the closed-form leave-one-out
    estimates, and `jf` is unchanged afterwards.
11. **No side effects:** after `snapshot`, `diagnostics` and `leave_one_out`, `Niter`,
    the `functionCalled` emission count, each model's `x` and parameters, and
    `result` are unchanged. A model-mode snapshot's `fit_intensity` path reproduces
    `result.chisqr` exactly, and an aborted or MCMC snapshot carries its label.
12. **Rescaling (§6.3):** off by default gives a bit-identical fit to today; on, two
    synthetic datasets with true error scales 1 and 2 recover s ≈ (1, 2) and Σν_i = N − p;
    with a shared term the transformed C is PSD and equals S C S; `inflate_only` never
    gives s < 1; the label, key field and `.jpar` round-trip are present; a rescaled vs
    unrescaled comparison is refused; logdet includes 2 Σ n_i ln s_i.
13. The existing suite stays green; any harness that compares report text is updated
    deliberately, with the diff reviewed.

## 11. Left to the builder

- The finite-difference step and the internal layout of `joint_fit_stats.py`.
- Exact wording of report lines, within the labels of §3.4.
- How `leave_one_out` copies a `JointFit` (spec round-trip or rebuild), as long as `jf`
  is untouched.
- Plan granularity.

## 12. Calls made in this document (for the user to check)

1. **The split (replaces roadmap Decision 2):** sequential (Cholesky-slice) terms for sums
   and the calibrated z; marginal terms for per-dataset quality and Fisher shares; the
   Mahalanobis split is not used. They coincide without shared terms.
2. **Per-dataset χ²/n, not χ²_red.**
3. **Log fits are first-order Gaussian likelihoods of ln y** and may show Δ(−logL);
   reverses a roadmap rule that predates the ln fix.
4. **Exact likelihood everywhere (user, 2026-11-11; replaces re-scoring):** every weighted
   objective is reported as −2 ln L with its log-determinant, split by dataset by the
   chain rule. `model`-mode fits are scored under their own C(θ̂). The fit gap and the
   distance to the exact optimum are reported. Optimising and sampling the exact
   likelihood is WS6. The Fisher trace term is omitted and footnoted.
5. **The key admits different models, PDBs, priors and methods**; data, windows, masks,
   scale, covariance, mode and dataset order must match.
6. **Per-dataset error rescaling is opt-in** (user, 2026-11-11), off by default, iterated,
   ν_i apportioned by points, labelled and keyed (§6.3). Two rescaled fits are compared
   each under its own s, with 2 Σ n_i ln s_i in the log-det; the N scales count in k.
7. **Compromise flag:** Δ_i > median + 3·MAD and Δ_i > 2.
8. **AIC/BIC kept, demoted, and corrected** to the known-error form.
9. **Anchor-and-verify and the sign-agreement check are deferred**; LOO is in.
10. **No GUI change**; the comparison panel keeps its text until WS4 (the swap is one
    call, so it can be pulled forward if you want it sooner).
11. Thresholds: q-range spread 10 %, heterogeneity ×3 (roadmap Decision 5, as proposed),
    both user-settable.

## 13. Implementation record

Implemented 2026-11-11 by `21-plan-ws2-reporting.md`, commits `c884afa..8078803` on `develop`
(nothing pushed). Code: `joint_fit_stats.py` (new), `Joint_Fit_Routines.py`; harnesses
`validate_joint_stats.py` (304 checks) and `validate_joint_diag.py` (258 checks).

### 13.1 Commits

- `64f1937` T1: `JointFit` quiet evaluation, row ranges, prior rows, last blocks and model blocks.
- `16d2e3c` T2: `JointSnapshot`, sequential/marginal split, exact likelihood, labels.
- `93d1e42` T3: comparability key.
- `1c10e91` T4: calibrated comparison, nesting, known-error AIC/BIC.
- `5bcbcc3` T4 fix: small-N headline, unweighted wording, fixed-value nesting.
- `1bc3dd4` T5: opt-in per-dataset error rescaling by sqrt(chi2_red).
- `3231ba3` T6: joint-fit diagnostics block (quality, heterogeneity, q-range, degeneracy).
- `5bf5298` T6 fix: quality and heterogeneity use unrescaled chi2/n on rescaled fits.
- `267acd6` T7: joint vs individual fits on one covariance.
- `5016e35` T7 fix: tests for a raising model, Log with a dropped row, rescaled fits, column mask.
- `1d984f5` T8: Fisher shares, model-reduction tension, distance to the exact optimum.
- `9b34e64` T8 fix: no optimum distance on emcee snapshots; ascii report tails in harnesses.
- `5867320` T9: leave-one-out refits on copies.
- `8078803` Final review fixes: ASCII report, labels, one evaluation per theta, abort and cheap diagnostics.

### 13.2 Tests

`validate_joint_stats.py`: Q1-Q6 (JointFit hooks), P1-P13 (split and exact likelihood), K1-K3
(comparability key), M1-M10 (comparison, nesting, IC), R1-R8 (rescaling). 304 checks, all pass.
`validate_joint_diag.py`: D1-D12 (diagnostics block), I1-I11 (joint vs individual), F1-F7 (Fisher,
tension, optimum), L1-L7 (leave-one-out), W1-W13 (final-review fixes: ASCII output, labels, one
evaluation per theta, abort, cheap mode, caching, rescale factor 0, restored outputs, stale
snapshot, LOO rows and settings, loop wording, tension under rescaling, Cholesky log-det).
258 checks, all pass. (Per-test check counts are not tabulated; the file totals are.)

### 13.3 Rulings (made during implementation)

- **R0.** Work on `develop` directly, as previous workstreams did; commits add explicit paths only.
- **R1.** The plan's P9 called `format_diagnostics`, which T6 creates. P9 now only checks that
  `per_dataset_objective` does not raise on an empty dataset; the `format_diagnostics` check moved to T6.
- **R2.** K1's "rescale on" case was built with `dataclasses.replace(snap, rescale={...})` because T5
  had not run; R5 retests the key on a real rescaled fit.
- **R3.** For the likelihood-ratio p-value with "A in B", the mirror uses nll2_A - nll2_B with
  delta k = b.nvarys - a.nvarys; the plan gave only "B in A".
- **R4.** Plan defect in D5: for y = (a+c)x + b the correlation is -1 and the softest direction is
  a - c, so the test asserts opposite signs (not "same sign") on a near-degenerate model, and
  "not positive definite (singular Hessian)" for the exact case.
- **R5.** A rescaled fit must not hide heterogeneity (spec 6.3): quality and heterogeneity use the
  unrescaled chi2/n.
- **R6.** With rescaling on, joint-vs-individual scores both sides under the unrescaled C_ii and says
  so, because the individual fit was never rescaled.
- **R7.** (Superseded by R9.) Keep unicode labels and wrap harness prints in `ascii()`.
- **R8.** `optimum_distance` is None for emcee snapshots: the point is posterior medians, not a
  fixed point.
- **R9.** Report text is ASCII. The pre-WS2 reports were pure ASCII, so unicode labels broke
  redirected stdout (cp1252). The section 3.4 labels keep their meaning with ASCII spellings:
  chi2, sigma_ln, sqrt(chi2_red), Delta, Sum d.
- **R10.** `JointFit.report_diagnostics` in {'full' (default), 'cheap'} gives users of expensive
  models an opt-out of the model-evaluating sections. The default keeps spec section 6 (every fit).

### 13.4 Deviations from the plan

- T8 F6 fixture: a per-point multiplicative term replaces the brief's rank-1 Sigma, whose
  perturbation gave delta of about 0.01 stderr (a vacuous test).
- T6 D5: see R4.
- The `ascii()` print sites added to `validate_covariance.py` and `validate_covariance_gui.py`
  in T8 are kept (harmless); the report itself is now ASCII.
- T5's commit trailer carries the implementing model's own attribution; history not rewritten.

### 13.5 Final review

Opus review of `c884afa..5867320`: ready with fixes. Important: I1 `snap.label` never rendered and a
false "Hessian failed" sentence on MCMC results; I2 non-ASCII report crashed redirected stdout;
I3 diagnostics cost (3 evaluations per theta), no abort handling, duplicate factorisations;
I4 rescale factor 0 (a dataset fitted exactly) crashed under nan_policy raise. Minors M1-M7
(restore model outputs, Cholesky log-det, stale-snapshot guard, LOO rows and settings, rescale-loop
wording, emcee + rescaling not a downgrade, docstring counts). All fixed in `8078803` with RED
evidence; the fixed-fit path is bit-identical to `c884afa` for six no-diagnostic cases. A scoped
re-review found no new Critical or Important issues. Remaining minors from it: cheap mode with
`individual_fits` still does one restore evaluation; a restore evaluation after a mid-loop abort;
`rescale_factors` notes only from round 0.

### 13.6 Full suite

`tests/run_all.py` (full, `QT_QPA_PLATFORM=offscreen`, cmc-py39) at `8078803`: **66 PASS, 0 FAIL, 0 SKIP, 0 TIMEOUT**, about 33 min. `tests/recovered/test_pdb_persist.py`, which failed once in an earlier `--quick` run, passed (3.9 s), so it is flaky rather than broken.

### 13.7 Notes and open items

- Report text is ASCII (R9); the 3.4 labels read chi2, sigma_ln, sqrt in place of the symbols.
- `JointFit.report_diagnostics` {'full','cheap'} exists (R10); code-settable only, no GUI control.
- Open: the finite-difference step may be noisy for XLayersPDB/XPR models; that noise decides the
  WS6 exact-optimiser question, and the distance-to-optimum on the bilayer bundle was not run.
- Open: leave-one-out deep-copies the models; this is untested for heavy or Julia-backed models.
- Open: the GUI `.jpar` save/load drops `#RescaleErrors` until WS4.
- Deferred minors that remain: `quiet()` restores only listed state (caches and output_params of
  XLayersPDB-style models are not restored); Q1 does not exercise the `lp.expr` branch; K1 index case
  does not pin index-before-data ordering; windows compared as exact floats; `rescale` dict not
  type-guarded; `_jitter_of` on unsymmetrised C; model_blocks substitution notes dropped in the
  snapshot; `worst_dataset` is the largest |d_i|; unweighted IC adds the prior to RSS;
  `joint_fit_stats.py` has mixed CRLF/LF line endings; R2 +-15% band is seed-picked; `_rescale_rows`
  recomputed per residual call; tension and Fisher tables truncate names to 12 characters; C is
  rebuilt several times per pass in uncached paths; uses private `cc._loading/_same_term`; LOO
  window rebuilt from x[imin]/x[imax] assumes sorted x, `fit_kws` not recorded per LOO row, no
  LOO regression for windows/masks/priors/rescale/Log beyond W10.
