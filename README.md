# TransportLab

**Inspect the plan, not just a distance-shaped number.** Dependency-free log-domain entropic optimal transport with full mass plans, explicit marginal convergence, zero-weight support, exact tiny uniform-assignment references and barycentric target means.

![Entropic plan, objectives and feasibility audit](assets/transport-audit.svg)

## Run offline

```sh
python -m unittest discover -s tests -v
python -m transportlab demo > /tmp/transport-demo.json
python examples/render_demo.py /tmp/transport-demo.json --check
python examples/render_pushforward.py /tmp/transport-demo.json --check
python -m transportlab solve examples/points.json --epsilon .5
python -m transportlab assignment examples/points.json
python -m transportlab push examples/points.json --epsilon 2 --time .5
```

Python3.10+, standard library only. No downloaded model, dataset, account, GPU or paid API. JSON input has exactly `source_points`, `target_points`, `source_weights`, `target_weights`. CLI `solve` returns a result even when the iteration budget is exhausted: **check `converged` and both residuals**. An unconverged result is not silently repaired or labeled optimal. `assignment` is limited to equal-size uniform weights, not general marginals.

## Mathematical contract

For nonnegative cost C and probability marginals a,b, the objective is

```
min_P sum(P_ij*C_ij) + epsilon * sum(P_ij*log(P_ij))
subject to P >= 0, P*1 = a, transpose(P)*1 = b
K_ij = exp(-C_ij/epsilon)
P_ij = exp(log_u_i - C_ij/epsilon + log_v_j)
log_u_i = log(a_i) - logsumexp_j(-C_ij/epsilon + log_v_j)
log_v_j = log(b_j) - logsumexp_i(-C_ij/epsilon + log_u_i)
```

Entropy is `-sum(P*log(P))` in nats, so `regularized_objective = transport_cost - epsilon*entropy`. This convention omits the optional `-epsilon*sum(P)` constant. Transport cost, entropy and regularized objective are separate outputs. Different epsilon values optimize different objectives, so their regularized scalars are not a shared ranking metric. Negative regularized objectives are expected, not negative transport costs.

Updates use log-sum-exp, not a pre-materialized kernel that can underflow entirely. Zero-weight rows/columns are removed during scaling and returned as exact zeros. A gauge recenter preserves `log_u+log_v`. Actual plan entries can still underflow to zero; their count is reported. The solver checks maximum absolute **row and column** marginal errors after each complete update, against tolerance1e-9 by default. Each full iteration ends with a target-column update, so small column error alone does not establish convergence.

Squared Euclidean ground costs are used by the point CLI and demo. They are not a metric. Finite-epsilon transport costs can have positive self-cost; this implementation is **not** the debiased Sinkhorn divergence, a Wasserstein metric, a differentiable training loss API or an exact general LP solver.

## Measured development fixture

Six fixed synthetic2D points; targets are a permuted translation(+0.6,-0.3). Uniform marginals. No learned embeddings, hidden labels, architecture/seed search, hyperparameter selection or downstream matching claim. All three epsilon settings and the nonconverged arm remain visible.

| epsilon | Converged at1e-9 | Iterations | Transport cost | Entropy, nats | Regularized objective | Max row error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | no | 5000 cap | 0.450027 | 1.792657 | 0.270761 | 3.452e-5 |
| 0.5 | yes | 576 | 0.522178 | 1.972472 | -0.464058 | 9.903e-10 |
| 2 | yes | 29 | 1.699807 | 3.079622 | -4.459437 | 8.688e-10 |

All column residuals are below3e-16 on this run. The epsilon0.1 arm exhausted the original budget; it is not an exact feasible plan or a certified optimum, even though its cost is near the assignment reference. No epsilon was replaced to hide that failure. Converged means marginal tolerance met, not globally certified arithmetic accuracy or a reported duality gap.

Enumerating all720 assignments gives the unregularized equal-uniform OT reference0.45; the independent product coupling costs4.227778. The assignment minimum is valid in this narrow scope because equal uniform couplings are scaled doubly-stochastic matrices and a linear minimum occurs at a permutation vertex. The exhaustive routine is not offered for unequal weights or general rectangular OT. Reference enumeration is factorial, not scalable. The entropic and unregularized objectives differ.

Self-transport at epsilon2 costs1.249807 despite identical supports. A separate unequal-marginal, one-iteration counterexample has row error0.4 while columns are already accurate. Both run in the demo. The heatmaps display every plan cell with shared scale0..1/6 and numeric masses; no top-k or edge threshold hides mass. Full plans, history, input points and barycentric means are in JSON.

## Barycentric maps are not samples

`barycentric(plan, target_points)` computes each conditional target mean using that row's **actual** mass. Zero-mass rows return `None`. A diffuse plan can map points between target support points: those means do not reproduce the target empirical distribution and are not guaranteed one-to-one correspondences. On an unconverged plan, requested source masses may differ from actual row masses; the function does not conceal that distinction.

## Mass pushforward versus mean-map collapse

![Full weighted interpolation atoms and barycentric means](assets/pushforward-audit.svg)

`pushforward(plan, source_points, target_points, time)` retains **every pair atom**, even zero-mass cells, at `(1-t)*x_i + t*y_j` with unchanged mass `P_ij`. It also returns one conditional mean per source, actual endpoint marginals, full and mean-map moments, and the within-source covariance. Moments normalize by the plan's actual positive total, not a requested marginal. At t=0 the full atoms reproduce the supplied row masses; at t=1 they reproduce the supplied column masses. Coincident atoms remain separate in JSON and are aggregated only for area-correct plotting.

The `push` CLI solves first and refuses a nonconverged plan, directing callers to `solve` for residual inspection. This refusal does not make an accepted plan exact: the solver's tolerance still applies. The API can audit any finite nonnegative supplied plan of the right shape, with positive total<=1e6, and **does not certify its feasibility** against any requested weights. Time must be finite in[0,1]; source/target point bounds remain unchanged. JSON keeps all atoms, masses, row means, covariance matrices and residuals.

For the same converged epsilon2 development coupling:

| t | Full-atom variance trace | Mean-map variance trace |
| ---: | ---: | ---: |
| 0 | 1.888889 | 1.888889 |
| 0.5 | 1.576437 | 1.319500 |
| 1 | 1.888889 | 0.861140 |

The endpoint target distribution is preserved by full weighted atoms using actual column sums, but not by the six barycentric means. The mean of the two representations agrees, while their covariance differs by the expected within-source covariance. This is the finite law of total covariance: `Cov(Z)=Cov(E[Z|source])+E[Cov(Z|source)]`. Because only the target is random within each source, this gap scales as t². It is an arithmetic identity, not evidence for optimal matching, latent geometry recovery or causal alignment.

Demo JSON audits both converged epsilon0.5/2 plans at t=0/0.5/1; all36 pair atoms and six conditional mean rows are retained per slice. Maximum covariance-identity residual is1.48e-16 on this run; that is checked-input arithmetic evidence, not a global bound. The spatial figure uses the more diffuse epsilon2 plan to make the distinction visible, with shared axes and circle area proportional to joint mass. No mass threshold, stochastic sampling or sampled path is implied. The original epsilon0.1 failed arm remains in the original solver tables; the push CLI explicitly refuses it rather than silently dropping its status.

Eight new tests check endpoint masses, interpolated means, deterministic zero conditional variance, independent-coupling collapse, the t² covariance gap, mass-scale invariance of normalized moments, zero rows, unchanged inputs, invalid plans/times and CLI convergence rejection. All original solver fixture fields and heatmaps remain unchanged.

## Audits and bounds

36 tests check the closed-form symmetric2x2 solution, direct non-log scaling reference, zero-cost independence, zero weights, underflowed kernels, shared cost shifts and simultaneous cost/epsilon scaling, rectangular/permuted marginals, explicit nonconvergence, positive self-cost, exhaustive assignment/endpoints, barycentric means, no mutation, strict bounds and CLI rejection. CI runs Python3.10-3.13 plus the full fixture and both reproduced SVGs on3.11.

Probability sums must be within1e-12 of1, then are explicitly normalized by their supplied sums; original sums and normalized weights are returned.1-64 rows per side, finite nonnegative costs<=1e6, epsilon[1e-4,1e4], iterations1-5000, tolerance[1e-12,1e-3], active_cells*max_iterations<=10,000,000. Point coordinates[-100,100]; two coordinates only. Assignment size1-8. Booleans/nonfinite values fail. Bounds are resource limits, not convergence guarantees. Costs depend on coordinate units and must scale with epsilon to preserve a plan. [DESIGN.md](DESIGN.md) gives implementation details and limits.

## Source credit

Independent educational implementation based on **Marco Cuturi (2013), [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/pdf/1306.0895)**, especially the transport polytope/entropy definitions in Section2 and penalized objective, scaling form and alternating updates in Section4, Eq.2-3/Algorithm1. Here `epsilon=1/lambda`. The log-domain formulation is an algebraic rewrite for numerical stability; no copied code, MNIST result, speedup, constrained-alpha metric theorem or benchmark reproduction is claimed.

MIT license.

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
