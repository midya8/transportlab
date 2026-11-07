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
