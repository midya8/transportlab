# Differentiating normalized marginals through fixed Sinkhorn updates

This extension follows the derivative of the computation that actually ran. It normalizes raw weights, performs exactly N log-domain row/column updates from zero scalings, differentiates each update and gauge recenter, and returns every intermediate plan and tangent. It does not differentiate a convergence test or pretend the final plan is an exact optimum.

The earlier `solve`, `audit`, `push`, `assignment`, `demo` and fixed-weight `sensitivity` APIs and their numerical outputs are unchanged. The new module accepts different raw-weight inputs and tighter bounds. It is not a replacement solver.

## Reproduce

```sh
python -m unittest discover -s tests -v
node --test tests/*.cjs
python examples/marginal_controls.py
python -m transportlab.marginal input.json > report.json
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/*.whl
bash examples/verify_marginal_wheel.sh /tmp/installed
```

The command takes an input file or `-` for stdin. A browser archive contains `input`, the normalized weights and tangents, raw totals, every iteration and the final iteration. Extract its `input` for a fresh native replay. The browser uses scalar dual-number arithmetic independently from the native hand-derived tangent loops. It does not read saved native reports.

No paid dependencies, external account, data download or learned embeddings. Node 22+ and Chrome are needed for the real browser checks. Python 3.10+ uses only the standard library at runtime.

## Input

Exactly nine fields are required:

```json
{
  "cost": [[0.2, 3], [1, 0.4]],
  "cost_direction": [[0, 0], [0, 0]],
  "source_weights": [9, 1],
  "source_direction": [1, -1],
  "target_weights": [1, 1],
  "target_direction": [0.2, -0.2],
  "epsilon": 0.5,
  "epsilon_direction": 0,
  "iterations": 1
}
```

The local perturbation is `w(s)=w+s*dw`, `C(s)=C+s*dC`, `epsilon(s)=epsilon+s*dEpsilon`. The module evaluates the derivative at `s=0`; it does not apply a finite edit. Raw source and target vectors normalize separately. A raw direction is not itself a normalized probability direction.

Zero raw entries require zero direction, so support is held fixed. Positive entries are bounded away from zero. Finite edits are the caller's responsibility: weights, costs and epsilon must stay inside the accepted domain. At a boundary, an algebraic derivative may only have an admissible one-sided interpretation, or no nonzero common step if different boundary directions conflict. No two-sided feasible perturbation is promised for every accepted formal direction.

## Normalization

For raw weights `w`, raw total `W=sum(w)`, raw total tangent `dW=sum(dw)`:

```text
a = w / W
da = (dw - a*dW) / W
```

Thus `sum(da)=0` in exact arithmetic. If `dw=k*w`, then `da=0`: rescaling a whole raw vector does not change the corresponding probability distribution. The normalization-only control uses nonzero raw directions but has zero normalized tangents, zero plan tangents and zero objective derivative. Its finite plan is still infeasible, so null sensitivity is not an optimality claim.

The JSON stores raw totals and raw-total tangents, as well as each raw, normalized and actual mass/tangent. No hidden projection or direction repair occurs.

## The computation and its derivative

The objective convention matches the original repository:

```text
F(P,C,epsilon) = sum(P*C) + epsilon*sum(P*log(P))
k = -C/epsilon
dk = -dC/epsilon + C*dEpsilon/epsilon²
u = log(a) - LSE(k+v), row-wise
v = log(b) - LSE(k+u), column-wise
```

For `LSE(z)`, the directional derivative is the softmax-weighted sum of `dz`. Therefore the new terms relative to fixed-marginal sensitivity are:

```text
du = da/a - softmax(k+v) · (dk+dv)
dv = db/b - softmax(k+u) · (dk+du)
```

Each full update ends with a column update. Then the mean of `u` is subtracted from `u` and added to `v`; the mean of `du` is handled the same way. This gauge recenter changes neither `u+k+v` nor its tangent. Every iteration is kept; there is no tolerance-based stop or omission of inconvenient early iterates.

```text
log(P) = u + k + v
dlog(P) = du + dk + dv
dP = P*dlog(P)
```

Active log masses remain in the report even when `exp(log(P))` underflows. Inactive rows/columns return zero masses/tangents and null potentials/log masses. A floating zero on active support is not a mathematical zero-support assertion.

## Direct terms, requested marginals and actual mass tangents

Let `R=row(P)`, `B=column(P)`, with corresponding derivatives `dR,dB`. For this repository's `p log(p)` convention define

```text
f = epsilon*u
g = epsilon*(v+1)
```

At the scaled plan, `C+epsilon*(log(P)+1)=f+g`. Applying the full chain rule to the finite computation gives

```text
direct = sum(P*dC) + dEpsilon*sum(P*log(P))
actual_mass_terms = f·dR + g·dB
full_tangent = direct + actual_mass_terms

requested_marginal_terms = f·da + g·db
envelope_candidate = direct + requested_marginal_terms
residual_tangent_terms = f·(dR-da) + g·(dB-db)
full_tangent = envelope_candidate + residual_tangent_terms
```

The last two identities are algebraic identities for the scaled finite iterate in real arithmetic. They do not establish that its potentials are optimal. The report keeps signed numerical identity residuals; it does not clamp a small residual to zero or label it a certified bound.

An optimum-envelope formula needs the relevant optimization assumptions and the optimum, not merely a finite scaled plan. The candidate deliberately contracts potentials with the **requested normalized** marginal tangents; the full computation instead responds through **actual** mass tangents. At a sufficiently converged numerical control these nearly agree, but that example does not give a general convergence or derivative-accuracy guarantee.

The common gauge transformation `f→f+c`, `g→g-c` leaves these contractions unchanged in exact arithmetic when the associated source/target total tangents agree. Normalized requested tangents each sum to zero; actual row/column tangents have the same total. The explicit gauge convention makes potential values and their derivatives reproducible, rather than identified separately by the transport plan.

## The early-weight control

The input shown above moves both normalized source and target marginals by `[0.1,-0.1]`, with fixed costs and epsilon.

| Fixed updates | Max row-mass error | Max row-tangent error | Full objective tangent | Candidate | Difference |
|---:|---:|---:|---:|---:|---:|
| 1 | .391902132 | .029003914 | .010665526 | -.019162331 | .029827857 |
| 10 | 1.70777e-8 | 1.23961e-7 | .060527918 | .060527510 | 4.08328e-7 |
| 50 | 1.11e-16 | 9.71e-17 | .060527519 | .060527519 | 2.08e-16 |

The one-update candidate even has the opposite sign from the full finite-computation derivative. At ten updates, a small primal row error coexists with a larger tangent error. These are arithmetic counterexamples and same-input iteration snapshots, not a universal algorithm ranking, theorem or recommendation to use fifty updates.

The cost-plus-weight-plus-epsilon control makes the distinction larger: after three updates its full tangent is about `.818104432` versus candidate `.195683813`, while row-tangent error is about `.212264622`. All directions and every history row are retained.

## Controls and independent checks

Twelve synthetic controls cover early/later weight edits, joint directions, normalization-only scaling, a zero-cost product coupling, one row, one column, zero support, active-entry underflow, rectangular plans, a uniform cost shift and simultaneous cost/epsilon scaling. No seed, epsilon or control is hidden to improve a result.

- Native tests use an independent Decimal80 positive-kernel matrix-scaling implementation with no log scalings, no gauge, no tangent formulas and no early stop. Central differences at `1e-20` check every plan tangent and both objective/transport tangents at 33 selected control update counts. Sixty fixed-seed random rectangular cases use independent Decimal80 central differences at `1e-18`.
- The zero-cost product plan has the separate closed-form derivative `dP=da*b+a*db`. Cost shift, simultaneous cost/epsilon scaling, raw-vector scaling, direction linearity and row/column permutation provide additional identities. Original fixed-marginal sensitivity agrees when raw weight directions are zero.
- Node compares complete native reports against independent scalar dual-number JavaScript at all 87 supplied updates, plus 80 generated cases. Numerical comparisons use absolute-plus-relative tolerance `2e-9*(1+abs(reference))`, not exact cross-runtime floating identity.
- Actual Chrome checks all 12 controls and all 87 selected update rows, 366 plan cells, displayed masses/tangents/log masses, 36 responsive layouts, physical slider and JSON keyboard edits, duplicate-key invalidation, clipboard failure/success and two completed Blob downloads replayed natively. Displayed rounded numbers have a separate `2e-8*(1+abs(reference))` tolerance.
- The full suite has 62 Python tests and six Node/actual-Chrome tests. All 46 earlier Python tests are retained. Installed replay imports the seven importable modules plus executes the main CLI, compares complete original demo/fixed-weight sensitivity outputs with source, then reproduces all 12 new controls, the new CLI and 12 invalid-input refusals outside the checkout.

Central differences check the specified smooth algebra on fixed support. At zero-cost boundaries their signed arithmetic extension is a derivative reference, not evidence that both finite perturbation signs are accepted by the public API. No formal verification, interval arithmetic or certified optimality is claimed.

## Bounds and costs

- 1-4 raw weights per side; each in `[0,100]`, each positive entry at least `1e-4`, positive total. Raw weight and cost directions are in `[-100,100]`; zero raw weights require zero direction.
- Cost cells in `[0,100]`; epsilon in `[0.01,100]`; epsilon direction in `[-100,100]`; fixed updates 1-500. Source and target dimensions may differ.
- Strict nine-field schema. Duplicate JSON keys, extra fields, booleans in numeric positions, nonfinite values, wrong shapes and unsupported bounds reject. CLI and browser input are capped at 64 KiB. Browser JSON nesting is capped at 64. Native nonfinite intermediate/output checks reject rather than serialize invalid numbers.
- Work is `O(N*n*m)` arithmetic operations for N updates. Full history intentionally stores `O(N*n*m)` values, including every plan/tangent and potential/tangent. No sparse, large-scale, streaming-memory or runtime claim.
- All arithmetic is binary64 except the independent test references. Underflow can erase tiny plan entries/tangents; cancellation and accumulated roundoff can affect potentials, normalized tangents and chain identities. A small residual is a diagnostic, not a certified error bound.

This is not implicit optimum differentiation, a derivative of adaptive stopping or rounding, support-changing weight differentiation, a debiased Sinkhorn divergence, a Wasserstein metric, causal influence, a training-quality result or a production API. Original numerical failures and nonconverged arms remain visible.

## Sources and convention

- Luise et al., [Differential Properties of Sinkhorn Approximation for Learning with Wasserstein Distance](https://proceedings.neurips.cc/paper_files/paper/2018/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf), NeurIPS2018: differentiated transport functionals and approximation distinctions.
- Eisenberger et al., [A Unified Framework for Implicit Sinkhorn Differentiation](https://openaccess.thecvf.com/content/CVPR2022/papers/Eisenberger_A_Unified_Framework_for_Implicit_Sinkhorn_Differentiation_CVPR_2022_paper.pdf), CVPR2022: implicit differentiation is a different method, not what this module implements.
- POT documentation, [Different gradient computations for regularized optimal transport](https://pythonot.github.io/auto_examples/backends/plot_Sinkhorn_gradients.html): autodiff/envelope distinctions. No POT dependency or code reuse.
- Nutz, [Introduction to Entropic Optimal Transport](https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf), and Peyre, [Computational Optimal Transport course slides](https://optimaltransport.github.io/slides-peyre/CourseOT.pdf): standard marginal constraints, scaling and potentials.
- [Non-asymptotic convergence bounds for Sinkhorn iterates and their gradients](https://arxiv.org/html/2304.06549v2): separate iterate/gradient convergence context. No theorem assumptions, rate or bound from this paper are claimed for these controls.

The module is an original implementation of standard normalization, chain-rule differentiation and Sinkhorn algebra. Its `sum(P log(P))` entropy convention omits other common constants, so the `g=epsilon*(v+1)` term is explicit. No source code copied, new theorem, source-author endorsement or benchmark reproduction.
