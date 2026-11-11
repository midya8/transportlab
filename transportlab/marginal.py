"""Differentiate a fixed Sinkhorn computation through raw weight normalization.

The original solver and fixed-marginal sensitivity API are deliberately separate.
No adaptive stop, implicit optimum derivative or rounding enters this module.
"""

import json
import math

FIELDS = {
    "cost",
    "cost_direction",
    "source_weights",
    "source_direction",
    "target_weights",
    "target_direction",
    "epsilon",
    "epsilon_direction",
    "iterations",
}


def _number(x, low, high, label):
    if type(x) not in (int, float) or not low <= x <= high or not math.isfinite(x):
        raise ValueError(label + " is outside its finite numeric bounds")
    return float(x)


def _weights(xs, ds, label):
    if (
        not isinstance(xs, list)
        or not 1 <= len(xs) <= 4
        or not isinstance(ds, list)
        or len(ds) != len(xs)
    ):
        raise ValueError(label + " needs 1..4 weights and same-length directions")
    w = [_number(x, 0, 100, label) for x in xs]
    d = [_number(x, -100, 100, label + " direction") for x in ds]
    if not any(w):
        raise ValueError(label + " total must be positive")
    if any(x and x < 1e-4 for x in w):
        raise ValueError("positive raw weights must be at least 1e-4")
    if any(x == 0 and v != 0 for x, v in zip(w, d)):
        raise ValueError("zero raw weights require zero direction: fixed support only")
    total = math.fsum(w)
    dt = math.fsum(d)
    a = [x / total for x in w]
    da = [(v - x * dt) / total for x, v in zip(a, d)]
    return w, d, a, da, total, dt


def parse(inp):
    if not isinstance(inp, dict) or set(inp) != FIELDS:
        raise ValueError("input needs exactly the nine documented fields")
    sw, sd, a, da, st, dst = _weights(
        inp["source_weights"], inp["source_direction"], "source"
    )
    tw, td, b, db, tt, dtt = _weights(
        inp["target_weights"], inp["target_direction"], "target"
    )
    n = len(a)
    m = len(b)

    def matrix(key, lo, hi):
        xs = inp[key]
        if (
            not isinstance(xs, list)
            or len(xs) != n
            or any(not isinstance(row, list) or len(row) != m for row in xs)
        ):
            raise ValueError(key + " matrix has wrong shape")
        return [[_number(x, lo, hi, key) for x in row] for row in xs]

    c = matrix("cost", 0, 100)
    dc = matrix("cost_direction", -100, 100)
    eps = _number(inp["epsilon"], 0.01, 100, "epsilon")
    de = _number(inp["epsilon_direction"], -100, 100, "epsilon direction")
    N = inp["iterations"]
    if type(N) is not int or not 1 <= N <= 500:
        raise ValueError("iterations must be integer 1..500")
    canonical = dict(
        cost=c,
        cost_direction=dc,
        source_weights=sw,
        source_direction=sd,
        target_weights=tw,
        target_direction=td,
        epsilon=eps,
        epsilon_direction=de,
        iterations=N,
    )
    return canonical, a, da, b, db, [st, tt], [dst, dtt]


def audit(inp):
    inp, a, da, b, db, sums, dsums = parse(inp)
    c = inp["cost"]
    dc = inp["cost_direction"]
    eps = inp["epsilon"]
    de = inp["epsilon_direction"]
    N = inp["iterations"]
    n = len(a)
    m = len(b)
    active_rows = [i for i in range(n) if a[i]]
    active_cols = [j for j in range(m) if b[j]]
    k = [[-c[i][j] / eps for j in active_cols] for i in active_rows]
    dk = [
        [-dc[i][j] / eps + c[i][j] * de / eps**2 for j in active_cols]
        for i in active_rows
    ]
    u = [0.0] * len(active_rows)
    v = [0.0] * len(active_cols)
    du = u[:]
    dv = v[:]
    history = []

    def lse(z, dz):
        peak = max(z)
        ex = [math.exp(x - peak) for x in z]
        total = math.fsum(ex)
        return peak + math.log(total), math.fsum(x * y for x, y in zip(ex, dz)) / total

    for iteration in range(1, N + 1):
        for i, ii in enumerate(active_rows):
            value, tangent = lse(
                [k[i][j] + v[j] for j in range(len(v))],
                [dk[i][j] + dv[j] for j in range(len(v))],
            )
            u[i] = math.log(a[ii]) - value
            du[i] = da[ii] / a[ii] - tangent
        for j, jj in enumerate(active_cols):
            value, tangent = lse(
                [k[i][j] + u[i] for i in range(len(u))],
                [dk[i][j] + du[i] for i in range(len(u))],
            )
            v[j] = math.log(b[jj]) - value
            dv[j] = db[jj] / b[jj] - tangent
        shift = math.fsum(u) / len(u)
        ds = math.fsum(du) / len(du)
        u = [x - shift for x in u]
        du = [x - ds for x in du]
        v = [x + shift for x in v]
        dv = [x + ds for x in dv]
        p = [[0.0] * m for _ in range(n)]
        dp = [[0.0] * m for _ in range(n)]
        lp = [[None] * m for _ in range(n)]
        dlp = [[None] * m for _ in range(n)]
        for i, ii in enumerate(active_rows):
            for j, jj in enumerate(active_cols):
                lp[ii][jj] = u[i] + k[i][j] + v[j]
                dlp[ii][jj] = du[i] + dk[i][j] + dv[j]
                p[ii][jj] = math.exp(lp[ii][jj])
                dp[ii][jj] = p[ii][jj] * dlp[ii][jj]
        rows = [math.fsum(row) for row in p]
        cols = [math.fsum(row[j] for row in p) for j in range(m)]
        drows = [math.fsum(row) for row in dp]
        dcols = [math.fsum(row[j] for row in dp) for j in range(m)]
        f = [None] * n
        g = [None] * m
        df = [None] * n
        dg = [None] * m
        for i, ii in enumerate(active_rows):
            f[ii] = eps * u[i]
            df[ii] = de * u[i] + eps * du[i]
        for j, jj in enumerate(active_cols):
            g[jj] = eps * (v[j] + 1)
            dg[jj] = de * (v[j] + 1) + eps * dv[j]
        cells = [(i, j) for i in active_rows for j in active_cols]
        transport = math.fsum(p[i][j] * c[i][j] for i, j in cells)
        dentropy = math.fsum(dp[i][j] * (lp[i][j] + 1) for i, j in cells)
        plogp = math.fsum(p[i][j] * lp[i][j] for i, j in cells)
        dt = math.fsum(dp[i][j] * c[i][j] + p[i][j] * dc[i][j] for i, j in cells)
        F = transport + eps * plogp
        dF = dt + de * plogp + eps * dentropy
        direct = math.fsum(p[i][j] * dc[i][j] for i, j in cells) + de * plogp
        actual_chain = math.fsum(f[i] * drows[i] for i in active_rows) + math.fsum(
            g[j] * dcols[j] for j in active_cols
        )
        requested_chain = math.fsum(f[i] * da[i] for i in active_rows) + math.fsum(
            g[j] * db[j] for j in active_cols
        )
        residual_chain = math.fsum(
            f[i] * (drows[i] - da[i]) for i in active_rows
        ) + math.fsum(g[j] * (dcols[j] - db[j]) for j in active_cols)
        entry = dict(
            iteration=iteration,
            plan=p,
            plan_tangent=dp,
            log_plan=lp,
            log_plan_tangent=dlp,
            actual_source=rows,
            actual_target=cols,
            actual_source_tangent=drows,
            actual_target_tangent=dcols,
            source_potential=f,
            target_potential=g,
            source_potential_tangent=df,
            target_potential_tangent=dg,
            transport_cost=transport,
            transport_tangent=dt,
            regularized_objective=F,
            full_tangent=dF,
            direct_terms=direct,
            requested_marginal_terms=requested_chain,
            actual_mass_terms=actual_chain,
            envelope_candidate=direct + requested_chain,
            residual_tangent_terms=residual_chain,
            chain_identity_signed_residual=dF - direct - actual_chain,
            envelope_identity_signed_residual=dF
            - direct
            - requested_chain
            - residual_chain,
            row_residual=max(abs(x - y) for x, y in zip(rows, a)),
            column_residual=max(abs(x - y) for x, y in zip(cols, b)),
            row_tangent_residual=max(abs(x - y) for x, y in zip(drows, da)),
            column_tangent_residual=max(abs(x - y) for x, y in zip(dcols, db)),
            underflowed_active_entries=sum(p[i][j] == 0 for i, j in cells),
        )
        history.append(entry)
    result = dict(
        version="transportlab_marginal_tangent_v1",
        input=inp,
        source=a,
        target=b,
        source_tangent=da,
        target_tangent=db,
        raw_totals=sums,
        raw_total_tangents=dsums,
        active_rows=active_rows,
        active_columns=active_cols,
        history=history,
        final=history[-1],
        semantics="Derivative of exactly N log row/column updates and raw-weight normalization on fixed support. No adaptive stop, implicit optimum, rounded plan or certified numerical bound.",
    )

    def finite(x):
        if type(x) is float and not math.isfinite(x):
            raise ValueError("nonfinite tangent result")
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        if isinstance(x, list):
            for v in x:
                finite(v)

    finite(result)
    return result


def main(argv=None):
    import argparse, sys

    p = argparse.ArgumentParser(
        description="Finite Sinkhorn derivatives through normalized raw weights"
    )
    p.add_argument("input")
    args = p.parse_args(argv)

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    try:
        if args.input == "-":
            text = sys.stdin.read(65537)
        else:
            with open(args.input, encoding="utf-8") as f:
                text = f.read(65537)
        if len(text.encode()) > 65536:
            raise ValueError("input exceeds 64 KiB")
        result = audit(json.loads(text, object_pairs_hook=unique))
        output = json.dumps(result, allow_nan=False, sort_keys=True, indent=2)
    except (ValueError, TypeError, OSError, OverflowError, RecursionError) as e:
        print("error: " + str(e), file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
