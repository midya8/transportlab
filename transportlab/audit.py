"""Separate finite-precision primal/dual audit, never a silent solver repair."""
import math
from .core import sinkhorn, weights, number


def round_plan(plan, source, target):
    """Row downscale, column downscale, then rank-one deficit fill (AWR2017)."""
    a,_=weights(source,'source');b,_=weights(target,'target')
    if not isinstance(plan,list) or len(plan)!=len(a):raise ValueError('plan row count mismatch')
    f=[]
    for row in plan:
        if not isinstance(row,list) or len(row)!=len(b):raise ValueError('plan width mismatch')
        v=[number(x,'plan mass') for x in row]
        if any(x<0 for x in v):raise ValueError('negative plan mass')
        f.append(v)
    if math.fsum(x for row in f for x in row)>1e6:raise ValueError('plan mass exceeds1e6')
    rows=[math.fsum(row) for row in f]
    x=[min(a[i]/r,1.) if r else 1. for i,r in enumerate(rows)]
    h=[[x[i]*v for v in row] for i,row in enumerate(f)]
    cols=[math.fsum(row[j] for row in h) for j in range(len(b))]
    y=[min(b[j]/c,1.) if c else 1. for j,c in enumerate(cols)]
    h=[[v*y[j] for j,v in enumerate(row)] for row in h]
    dr=[a[i]-math.fsum(row) for i,row in enumerate(h)]
    dc=[b[j]-math.fsum(row[j] for row in h) for j in range(len(b))]
    # Subtraction can produce negative ulps after downscaling. Disclose clipping.
    er=[max(0.,v) for v in dr];ec=[max(0.,v) for v in dc];missing=math.fsum(er)
    g=[[v+(er[i]*(ec[j]/missing) if missing else 0.) for j,v in enumerate(row)] for i,row in enumerate(h)]
    rr=[math.fsum(row) for row in g];cc=[math.fsum(row[j] for row in g) for j in range(len(b))]
    return {'plan':g,'row_scale':x,'column_scale':y,'row_deficits_before_clipping':dr,
            'column_deficits_before_clipping':dc,'clipped_negative_deficit_count':sum(v<0 for v in dr+dc),
            'deficit_total_mismatch':math.fsum(er)-math.fsum(ec),'rank_one_added_mass':math.fsum(ec) if missing else 0.,
            'row_max_abs_residual':max(abs(u-v) for u,v in zip(rr,a)),
            'column_max_abs_residual':max(abs(u-v) for u,v in zip(cc,b)),
            'l1_plan_change':math.fsum(abs(g[i][j]-f[i][j]) for i in range(len(a)) for j in range(len(b))),
            'semantics':'separate rounded plan; row/column downscales and rank-one fill; floating-point marginal residuals are reported'}


def audit_transport(cost,source,target,epsilon=.5,max_iterations=5000,tolerance=1e-9):
    raw=sinkhorn(cost,source,target,epsilon,max_iterations,tolerance,include_scaling=True)
    a,b=raw['source'],raw['target'];eps=raw['epsilon'];sc=raw['log_scaling']
    rows=[i for i,v in enumerate(a) if v];cols=[j for j,v in enumerate(b) if v]
    # For h(p)=eps*p*log(p), h*(s)=eps*exp(s/eps-1).
    f=[eps*v if v is not None else None for v in sc['source']]
    g=[eps*(v+1) if v is not None else None for v in sc['target']]
    conjugate=math.fsum(math.exp((f[i]+g[j]-cost[i][j])/eps-1) for i in rows for j in cols)
    dual=math.fsum(a[i]*f[i] for i in rows)+math.fsum(b[j]*g[j] for j in cols)-eps*conjugate
    rounded=round_plan(raw['plan'],a,b);p=rounded['plan']
    transport=math.fsum(p[i][j]*cost[i][j] for i in range(len(a)) for j in range(len(b)))
    entropy=-math.fsum(v*math.log(v) for row in p for v in row if v)
    primal=transport-eps*entropy
    gap=primal-dual
    if not all(math.isfinite(v) for v in [dual,primal,gap]):raise ValueError('nonfinite objective audit')
    rounded.update({'transport_cost':transport,'entropy_nats':entropy,'regularized_objective':primal})
    return {'raw_solver':raw,'rounded':rounded,'dual_source_potential':f,'dual_target_potential':g,
            'dual_value':dual,'dual_exponential_mass':conjugate,
            'signed_numerical_gap':gap,'raw_objective_minus_dual':raw['regularized_objective']-dual,
            'gap_semantics':'rounded regularized objective minus entropic dual; signed, not clipped; numerical audit, not an interval-certified bound or unregularized OT gap',
            'zero_marginal_semantics':'dual evaluated only on positive marginal rows/columns; zero support potentials are null'}
