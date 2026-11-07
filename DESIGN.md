# Transport, entropy and numerical limits

## Scaling direction

Rows are source, columns target. Positive marginal subsets form the active rectangle. With epsilon>0, a finite nonnegative cost matrix and strictly positive active weights, the mathematical entropy-regularized optimum has diagonal-scaling form diag(u)*exp(-C/epsilon)*diag(v). The iterative solver starts log_u/log_v0, updates all rows then all columns. The common gauge shift mean(log_u) is removed from u and added to v after every iteration, preserving the plan. We materialize the plan after each complete cycle and check both marginals. This is O(iterations*n*m), with memory O(n*m). The cell-update budget is a preflight product, not a FLOP count or wall-clock estimate; logsumexp and materialization add work.

The objective convention is C dot P + epsilon sum(P log P), with zero terms defined0. Entropy uses the stored pre-exponentiation logP, which is finite on active entries; if exp(logP) underflows, its term contributes0 and the count is explicit. We do not compute a duality gap, round a nearly feasible plan into a feasible one, enforce an arbitrary sparsity threshold or normalize the entire plan afterward. In particular, low cost before feasibility is not a valid optimality bound. A convergence flag only says returned marginal errors meet the requested absolute tolerance.

## Scope and validation

Probability vectors are finite nonnegative lists of length1-64. Supplied sums must agree with1 within1e-12; we divide by those actual sums and return both original sums and used vectors. This is declared small numerical normalization, not support for unbalanced mass. Exact-zero weights preserve zero rows/columns and do not enter log. Costs must be a matching rectangular list of lists, finite in[0,1e6]. Epsilon[1e-4,1e4], iterations1-5000, tolerance[1e-12,1e-3]; active_cells*max_iterations<=10million. All these checks finish before solver iteration. Coordinates are finite numeric2D lists in[-100,100]. Booleans fail.

Large C/epsilon can lead to slow convergence, cancellation in dual log sums and underflowed final entries even though the raw kernel is never formed. A fixed iteration budget need not converge. Numeric validation is not a certificate of feasibility or optimality; the measured residuals are authoritative. No epsilon continuation, warm start, accelerator, adaptive precision, implicit differentiation or GPU backend. Demo epsilon0.1 actually fails its original1e-9 tolerance after5000 updates; it remains in all outputs and figures. No setting sweep was used to find a replacement convergent low-epsilon result.

## Assignment reference

The exhaustive reference accepts a square cost matrix of size1-8 and implicitly uniform mass1/n on both sides. Every permutation is checked with cost averaged by n. Strict-less comparison retains the first lexicographic minimum when costs tie. This is a reference for the unregularized objective only. It is not a solver for nonuniform or rectangular marginals, and its plan is not generally the entropic optimum. CLI verifies equal uniform supplied weights before invoking it. The scaled Birkhoff polytope ensures a permutation minimum in this case; this reduction does not hold as implemented for arbitrary weights.

## What a returned map means

Barycentric output divides sum_j P_ij*y_j by actual sum_j P_ij, returning None for empty rows. It gives a mean, not a discrete transport sample or hard matching. A source mass can split over several columns. No claimed geometry recovery, causal alignment, learned representation, privacy, fair allocation, calibrated confidence or downstream accuracy. Squared costs depend on units; neither the squared ground cost nor this finite-epsilon scalar is presented as a metric or debiased Sinkhorn divergence. Cost/epsilon scaling together preserves the mathematical plan and is numerically tested.

## Full weighted pushforward

Each supplied joint cell P_ij becomes one atom z_ij=(1-t)x_i+t*y_j. Atom mass is unchanged, including zeros, not divided by the total. Actual row/column sums and total are returned. Moment expectations divide by that actual total; no normalization is written back to atom masses. Barycentric atoms have row-sum mass and the conditional row mean, or None at zero row mass. Covariance is accumulated from centered coordinates, avoiding second-moment subtraction. The expected within-source covariance is centered on each row's mean, averaged with normalized joint mass. The identity residual is max-coordinate absolute difference Cov(full)-Cov(means)-E[Cov(within)]. The gap is positive semidefinite mathematically, not a floating-point certificate; tiny signed roundoff residuals remain visible.

Only t in[0,1] is accepted. All point bounds and finite mass/shape checks run before atom construction. Total must be positive and<=1e6. There are at most4096 pair atoms; no hidden threshold, coalescing, randomness or trajectory integration. Exact coincident points are coalesced by the plotting script solely so symbol area reflects total overlapping mass; JSON preserves all cells. CLI push requires the immediately computed solver's convergence flag and carries its entire result alongside the pushforward. Arbitrary API plans need not have unit mass or requested marginals. Full atoms at endpoints reflect actual masses, never an unverified requested distribution. This is an algebraic interpolation under a fixed coupling, not a learned flow, geodesic certificate, sample generator, hard assignment or proof of OT optimality.

## Separate primal and dual numerical audit

A raw objective from a plan with wrong marginals is not a feasible primal bound. `audit_transport` and the `audit` CLI retain that raw result, expose opt-in log scaling, and return a **separate** rounded plan plus entropic dual evaluation. No raw entries, convergence flags or iteration counts are overwritten.

```sh
python -m transportlab audit examples/points.json --epsilon .1
python -m transportlab audit examples/points.json --epsilon .5 --max-iterations 1
```

For the same six-point development fixture:

| Epsilon | Raw converged | Raw objective | Dual value | Rounded objective | Signed numerical gap | L1 plan change |
|---|---|---|---|---|---|---|
| 0.1 | False | 0.270760837 | 0.270814894 | 0.271216935 | 0.000402 | 0.0001 |
| 0.5 | True | -0.464058401 | -0.464058395 | -0.464058395 | 2.09e-10 | 4.42e-09 |
| 2 | True | -4.459437262 | -4.459437257 | -4.459437257 | 0 | 3.53e-09 |

These are different regularized problems, not a shared epsilon ranking. Rounded row/column residuals are at most2.78e-17 on these inputs. The epsilon0.1 raw value lies **below** its dual because its marginals are wrong. Its failure remains a failure; separate rounding changes the plan by L1≈0.000100358. A one-iteration unequal-marginal counterexample is more severe: raw objective−0.006931472, dual0.001857427, rounded objective1.590566516, signed gap1.588709090 and L1 change0.8. That arm still has raw row error0.4. The demo asserts both retained failure and small rounded residuals.

For the repository convention `sum(P*C)+epsilon*sum(P*log(P))`, the dual on active positive-marginal rows/columns is:

```text
D(f,g) = a.f + b.g - epsilon * sum(exp((f_i+g_j-C_ij)/epsilon - 1))
f = epsilon * log_u; g = epsilon * (log_v + 1)
signed_numerical_gap = rounded_regularized_objective - D(f,g)
```

Weak duality gives a lower dual and upper feasible primal in exact arithmetic. **These outputs are floating-point numerical audits, not interval-certified bounds.** Gap stays signed, including negative roundoff; a displayed zero is not a proof of exact optimality. Rounding is row downscaling, column downscaling, then a rank-one nonnegative deficit fill. Tiny negative subtraction deficits are clipped to zero with count and pre-clipping values returned, alongside both marginal residuals, deficit-total mismatch, scales, added mass and L1 change. Zero rows/columns are preserved; dual potentials there are null and excluded. Exponential underflow and potential cancellation can affect numerical objectives. Nothing here bounds unregularized OT error or makes the original plan feasible.

`round_plan` accepts finite nonnegative rectangular matrices with total mass at most1e6, including all-zero input, using the existing1..64 marginal bounds. It does not mutate inputs. `audit_transport` keeps existing solver cost/epsilon/work bounds. Tests include an independent closed-form2x2 optimum, a zero-cost unequal-marginal optimum, gauge/cost-shift invariance, underflow and zero support, independent feasible-plan weak-duality comparisons,30 deterministic randomized rectangular rounding/capped-solve checks and CLI errors. **36 tests total.**

Rounding attribution: independently implemented from Algorithm2 in [Altschuler, Weed and Rigollet (2017), Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration](https://proceedings.neurips.cc/paper_files/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf). The dual is derived from the convex conjugate of `epsilon*p*log(p)` with this repository's constant convention. No reference code copied; no paper runtime, theorem guarantee, Greenkhorn implementation or benchmark result claimed.
