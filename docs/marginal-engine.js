"use strict";
// Scalar forward automatic differentiation, independent of native tangent loops.
(function (root) {
  class D {
    constructor(v, d = 0) {
      this.v = v;
      this.d = d;
    }
    add(q) {
      return new D(this.v + q.v, this.d + q.d);
    }
    sub(q) {
      return new D(this.v - q.v, this.d - q.d);
    }
    mul(q) {
      return new D(this.v * q.v, this.d * q.v + this.v * q.d);
    }
    div(q) {
      return new D(this.v / q.v, (this.d - (q.d * this.v) / q.v) / q.v);
    }
    log() {
      return new D(Math.log(this.v), this.d / this.v);
    }
    exp() {
      const v = Math.exp(this.v);
      return new D(v, v * this.d);
    }
  }
  const constant = (x) => new D(x),
    sum = (xs) => xs.reduce((s, v) => s.add(v), constant(0)),
    range = (n) => Array.from({ length: n }, (_, i) => i);
  const keys = (o, ks) =>
    o &&
    typeof o === "object" &&
    !Array.isArray(o) &&
    Object.keys(o).sort().join("|") === ks.slice().sort().join("|");
  function number(x, lo, hi, label) {
    if (typeof x !== "number" || !Number.isFinite(x) || x < lo || x > hi)
      throw Error(label + " is outside finite numeric bounds");
    return x;
  }
  function parse(inp) {
    if (
      !keys(inp, [
        "cost",
        "cost_direction",
        "source_weights",
        "source_direction",
        "target_weights",
        "target_direction",
        "epsilon",
        "epsilon_direction",
        "iterations",
      ])
    )
      throw Error("Input requires exactly the nine documented fields");
    function weights(xs, ds, label) {
      if (
        !Array.isArray(xs) ||
        xs.length < 1 ||
        xs.length > 4 ||
        !Array.isArray(ds) ||
        ds.length !== xs.length
      )
        throw Error(label + " needs1..4 weights with matching directions");
      const raw = xs.map(
        (v, i) =>
          new D(
            number(v, 0, 100, label),
            number(ds[i], -100, 100, label + " direction"),
          ),
      );
      if (!raw.some((x) => x.v)) throw Error("Raw total must be positive");
      if (raw.some((x) => x.v > 0 && x.v < 1e-4))
        throw Error("Positive raw weights must be at least1e-4");
      if (raw.some((x) => x.v === 0 && x.d !== 0))
        throw Error(
          "Zero raw weights require zero direction: fixed support only",
        );
      const total = sum(raw);
      return [raw.map((x) => x.div(total)), total];
    }
    const [a, as] = weights(inp.source_weights, inp.source_direction, "source"),
      [b, bs] = weights(inp.target_weights, inp.target_direction, "target"),
      n = a.length,
      m = b.length;
    function matrix(k, lo, hi) {
      const rows = inp[k];
      if (
        !Array.isArray(rows) ||
        rows.length !== n ||
        rows.some((r) => !Array.isArray(r) || r.length !== m)
      )
        throw Error(k + " matrix has wrong shape");
      return rows.map((row) => row.map((x) => number(x, lo, hi, k)));
    }
    const c = matrix("cost", 0, 100),
      dc = matrix("cost_direction", -100, 100),
      eps = number(inp.epsilon, 0.01, 100, "epsilon"),
      de = number(inp.epsilon_direction, -100, 100, "epsilon direction");
    if (
      !Number.isInteger(inp.iterations) ||
      inp.iterations < 1 ||
      inp.iterations > 500
    )
      throw Error("Iterations must be integer1..500");
    return [
      {
        cost: c,
        cost_direction: dc,
        source_weights: inp.source_weights.slice(),
        source_direction: inp.source_direction.slice(),
        target_weights: inp.target_weights.slice(),
        target_direction: inp.target_direction.slice(),
        epsilon: eps,
        epsilon_direction: de,
        iterations: inp.iterations,
      },
      a,
      b,
      as,
      bs,
    ];
  }
  function audit(input) {
    const [inp, a, b, as, bs] = parse(input),
      n = a.length,
      m = b.length,
      rows = range(n).filter((i) => a[i].v > 0),
      cols = range(m).filter((j) => b[j].v > 0),
      eps = new D(inp.epsilon, inp.epsilon_direction),
      C = inp.cost.map((row, i) =>
        row.map((v, j) => new D(v, inp.cost_direction[i][j])),
      ),
      K = rows.map((i) => cols.map((j) => constant(0).sub(C[i][j].div(eps))));
    const lse = (xs) => {
      const peak = Math.max(...xs.map((x) => x.v));
      return sum(xs.map((x) => x.sub(constant(peak)).exp()))
        .log()
        .add(constant(peak));
    };
    let u = rows.map(() => constant(0)),
      v = cols.map(() => constant(0));
    const history = [];
    for (let step = 1; step <= inp.iterations; step++) {
      u = rows.map((i, ii) =>
        a[i].log().sub(lse(cols.map((j, jj) => K[ii][jj].add(v[jj])))),
      );
      v = cols.map((j, jj) =>
        b[j].log().sub(lse(rows.map((i, ii) => K[ii][jj].add(u[ii])))),
      );
      const shift = sum(u).div(constant(u.length));
      u = u.map((x) => x.sub(shift));
      v = v.map((x) => x.add(shift));
      const lp = range(n).map(() => range(m).map(() => null)),
        P = range(n).map(() => range(m).map(() => constant(0)));
      rows.forEach((i, ii) =>
        cols.forEach((j, jj) => {
          lp[i][j] = u[ii].add(K[ii][jj]).add(v[jj]);
          P[i][j] = lp[i][j].exp();
        }),
      );
      const rowmass = P.map(sum),
        colmass = range(m).map((j) => sum(P.map((row) => row[j]))),
        f = Array(n).fill(null),
        g = Array(m).fill(null);
      rows.forEach((i, ii) => (f[i] = eps.mul(u[ii])));
      cols.forEach((j, jj) => (g[j] = eps.mul(v[jj].add(constant(1)))));
      const cells = rows.flatMap((i) => cols.map((j) => [i, j])),
        transport = sum(cells.map(([i, j]) => P[i][j].mul(C[i][j]))),
        plogp = sum(cells.map(([i, j]) => P[i][j].mul(lp[i][j]))),
        objective = transport.add(eps.mul(plogp));
      const direct =
        cells.reduce(
          (s, [i, j]) => s + P[i][j].v * inp.cost_direction[i][j],
          0,
        ) +
        inp.epsilon_direction * plogp.v;
      const actual =
          rows.reduce((s, i) => s + f[i].v * rowmass[i].d, 0) +
          cols.reduce((s, j) => s + g[j].v * colmass[j].d, 0),
        requested =
          rows.reduce((s, i) => s + f[i].v * a[i].d, 0) +
          cols.reduce((s, j) => s + g[j].v * b[j].d, 0),
        residual =
          rows.reduce((s, i) => s + f[i].v * (rowmass[i].d - a[i].d), 0) +
          cols.reduce((s, j) => s + g[j].v * (colmass[j].d - b[j].d), 0);
      const vals = (xs) => xs.map((x) => (x === null ? null : x.v)),
        ders = (xs) => xs.map((x) => (x === null ? null : x.d)),
        maxAbs = (xs) => Math.max(...xs.map(Math.abs));
      history.push({
        iteration: step,
        plan: P.map(vals),
        plan_tangent: P.map(ders),
        log_plan: lp.map(vals),
        log_plan_tangent: lp.map(ders),
        actual_source: vals(rowmass),
        actual_target: vals(colmass),
        actual_source_tangent: ders(rowmass),
        actual_target_tangent: ders(colmass),
        source_potential: vals(f),
        target_potential: vals(g),
        source_potential_tangent: ders(f),
        target_potential_tangent: ders(g),
        transport_cost: transport.v,
        transport_tangent: transport.d,
        regularized_objective: objective.v,
        full_tangent: objective.d,
        direct_terms: direct,
        requested_marginal_terms: requested,
        actual_mass_terms: actual,
        envelope_candidate: direct + requested,
        residual_tangent_terms: residual,
        chain_identity_signed_residual: objective.d - direct - actual,
        envelope_identity_signed_residual:
          objective.d - direct - requested - residual,
        row_residual: maxAbs(rowmass.map((x, i) => x.v - a[i].v)),
        column_residual: maxAbs(colmass.map((x, j) => x.v - b[j].v)),
        row_tangent_residual: maxAbs(rowmass.map((x, i) => x.d - a[i].d)),
        column_tangent_residual: maxAbs(colmass.map((x, j) => x.d - b[j].d)),
        underflowed_active_entries: cells.filter(([i, j]) => P[i][j].v === 0)
          .length,
      });
    }
    const result = {
      version: "transportlab_marginal_tangent_v1",
      input: inp,
      source: a.map((x) => x.v),
      target: b.map((x) => x.v),
      source_tangent: a.map((x) => x.d),
      target_tangent: b.map((x) => x.d),
      raw_totals: [as.v, bs.v],
      raw_total_tangents: [as.d, bs.d],
      active_rows: rows,
      active_columns: cols,
      history,
      final: history.at(-1),
      semantics:
        "Derivative of exactly N log row/column updates and raw-weight normalization on fixed support. No adaptive stop, implicit optimum, rounded plan or certified numerical bound.",
    };
    function finite(x) {
      if (typeof x === "number" && !Number.isFinite(x))
        throw Error("Nonfinite tangent result");
      if (x && typeof x === "object")
        for (const v of Object.values(x)) finite(v);
    }
    finite(result);
    return result;
  }
  const api = { audit };
  if (typeof module !== "undefined") module.exports = api;
  root.MARGINAL_ENGINE = api;
})(globalThis);
