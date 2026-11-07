"""Log-domain entropic transport, with explicit convergence and objective semantics."""
import itertools
import math


def integer(x,lo,hi,name):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(f'{name} must be integer in [{lo},{hi}]')
    return x


def number(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(name+' must be finite')
    return float(x)


def weights(x,name):
    if not isinstance(x,list) or not 1<=len(x)<=64:raise ValueError(name+' needs 1..64 weights')
    v=[number(z,name) for z in x];total=math.fsum(v)
    if any(z<0 for z in v) or abs(total-1)>1e-12:raise ValueError(name+' must be nonnegative and sum to1 within1e-12')
    return [z/total for z in v],total


def matrix(cost,n,m):
    if not isinstance(cost,list) or len(cost)!=n:raise ValueError('cost row count mismatch')
    out=[]
    for row in cost:
        if not isinstance(row,list) or len(row)!=m:raise ValueError('cost column count mismatch')
        values=[number(z,'cost') for z in row]
        if any(not 0<=z<=1e6 for z in values):raise ValueError('cost outside [0,1e6]')
        out.append(values)
    return out


def lse(values):
    high=max(values)
    return high+math.log(math.fsum(math.exp(v-high) for v in values))


def sinkhorn(cost,source,target,epsilon=.5,max_iterations=5000,tolerance=1e-9,include_scaling=False):
    if type(include_scaling) is not bool:raise ValueError('include_scaling must be bool')
    a,asum=weights(source,'source');b,bsum=weights(target,'target');c=matrix(cost,len(a),len(b))
    eps=number(epsilon,'epsilon');tol=number(tolerance,'tolerance')
    if not 1e-4<=eps<=1e4:raise ValueError('epsilon outside [1e-4,1e4]')
    if not 1e-12<=tol<=1e-3:raise ValueError('tolerance outside [1e-12,1e-3]')
    steps=integer(max_iterations,1,5000,'max_iterations')
    rows=[i for i,v in enumerate(a) if v>0];cols=[j for j,v in enumerate(b) if v>0]
    if len(rows)*len(cols)*steps>10000000:raise ValueError('active cells times max_iterations exceeds10000000')
    loga=[math.log(a[i]) for i in rows];logb=[math.log(b[j]) for j in cols]
    kernel=[[-c[i][j]/eps for j in cols] for i in rows]
    u=[0.]*len(rows);v=[0.]*len(cols);history=[]
    for step in range(1,steps+1):
        u=[loga[i]-lse([kernel[i][j]+v[j] for j in range(len(cols))]) for i in range(len(rows))]
        v=[logb[j]-lse([kernel[i][j]+u[i] for i in range(len(rows))]) for j in range(len(cols))]
        shift=math.fsum(u)/len(u);u=[z-shift for z in u];v=[z+shift for z in v]
        logp=[[u[i]+kernel[i][j]+v[j] for j in range(len(cols))] for i in range(len(rows))]
        p=[[math.exp(z) for z in row] for row in logp]
        rr=[math.fsum(row) for row in p];cc=[math.fsum(row[j] for row in p) for j in range(len(cols))]
        er=max(abs(rr[i]-a[k]) for i,k in enumerate(rows));ec=max(abs(cc[j]-b[k]) for j,k in enumerate(cols));res=max(er,ec)
        if step==1 or step%100==0 or res<=tol or step==steps:history.append({'iteration':step,'row_max_abs_residual':er,'column_max_abs_residual':ec})
        if res<=tol:break
    plan=[[0.]*len(b) for _ in a];transport=entropy=0.
    for i,ii in enumerate(rows):
        for j,jj in enumerate(cols):
            value=p[i][j];plan[ii][jj]=value;transport+=value*c[ii][jj]
            entropy-=value*logp[i][j]
    if not all(math.isfinite(z) for z in [transport,entropy,res]):raise ValueError('nonfinite solver result')
    result={'plan':plan,'source':a,'target':b,'input_weight_sums':[asum,bsum],
            'epsilon':eps,'iterations':step,'converged':res<=tol,'tolerance':tol,
            'row_max_abs_residual':er,'column_max_abs_residual':ec,'total_mass':math.fsum(rr),
            'transport_cost':transport,'entropy_nats':entropy,
            'regularized_objective':transport-eps*entropy,'history':history,
            'objective_convention':'sum(P*C) + epsilon*sum(P*log(P)); zero terms contribute0',
            'underflowed_active_entries':sum(z==0 for row in p for z in row),
            'active_shape':[len(rows),len(cols)],'cell_update_budget':len(rows)*len(cols)*steps}
    if include_scaling:
        lu=[None]*len(a);lv=[None]*len(b)
        for i,k in enumerate(rows):lu[k]=u[i]
        for j,k in enumerate(cols):lv[k]=v[j]
        result['log_scaling']={'source':lu,'target':lv}
    return result


def points(x):
    if not isinstance(x,list) or not 1<=len(x)<=64:raise ValueError('need1..64 points')
    out=[]
    for row in x:
        if not isinstance(row,list) or len(row)!=2:raise ValueError('points need2 coordinates')
        v=[number(z,'coordinate') for z in row]
        if any(abs(z)>100 for z in v):raise ValueError('coordinates outside[-100,100]')
        out.append(v)
    return out


def squared_cost(source,target):
    x,y=points(source),points(target)
    return [[sum((u-v)**2 for u,v in zip(a,b)) for b in y] for a in x]


def barycentric(plan,target):
    y=points(target)
    if not isinstance(plan,list) or not 1<=len(plan)<=64:raise ValueError('plan needs1..64 rows')
    mapped=[];row_mass=[]
    for row in plan:
        if not isinstance(row,list) or len(row)!=len(y):raise ValueError('plan width mismatch')
        values=[number(z,'plan mass') for z in row]
        if any(z<0 for z in values):raise ValueError('negative plan mass')
        mass=math.fsum(values);row_mass.append(mass)
        mapped.append([math.fsum(v*p[k] for v,p in zip(values,y))/mass for k in range(2)] if mass else None)
    return {'points':mapped,'actual_row_mass':row_mass,'semantics':'conditional target mean by actual plan row mass; not transported samples or a discrete target distribution'}


def exact_uniform_assignment(cost):
    if not isinstance(cost,list):raise ValueError('cost must be matrix')
    n=integer(len(cost),1,8,'assignment size');c=matrix(cost,n,n)
    best=None;count=0
    for perm in itertools.permutations(range(n)):
        value=math.fsum(c[i][j] for i,j in enumerate(perm))/n;count+=1
        if best is None or value<best[0]:best=value,perm
    plan=[[float(best[1][i]==j)/n for j in range(n)] for i in range(n)]
    return {'transport_cost':best[0],'permutation':list(best[1]),'plan':plan,'permutations_checked':count,
            'scope':'unregularized OT reference for equal-size uniform weights only; lexicographically first optimal assignment'}


def pushforward(plan,source_points,target_points,time=.5):
    """All atoms of (1-t)*X+t*Y under the supplied joint masses.

    This audits the supplied plan, not feasibility against requested marginals.
    Moments normalize by actual positive total mass; atom masses are unchanged.
    """
    x,y=points(source_points),points(target_points);t=number(time,'time')
    if not 0<=t<=1:raise ValueError('time outside[0,1]')
    if not isinstance(plan,list) or len(plan)!=len(x):raise ValueError('plan/source shape mismatch')
    masses=[]
    for row in plan:
        if not isinstance(row,list) or len(row)!=len(y):raise ValueError('plan/target shape mismatch')
        values=[number(v,'plan mass') for v in row]
        if any(v<0 for v in values):raise ValueError('negative plan mass')
        masses.append(values)
    total=math.fsum(v for row in masses for v in row)
    if not math.isfinite(total) or not 0<total<=1e6:raise ValueError('plan total mass must be in(0,1e6]')
    rows=[math.fsum(row) for row in masses]
    cols=[math.fsum(row[j] for row in masses) for j in range(len(y))]
    atoms=[];means=[];bary_atoms=[]
    for i,row in enumerate(masses):
        mapped=[]
        for j,mass in enumerate(row):
            point=[(1-t)*x[i][k]+t*y[j][k] for k in range(2)]
            atoms.append({'source_index':i,'target_index':j,'point':point,'mass':mass})
            mapped.append(point)
        mean=[math.fsum(v*p[k] for v,p in zip(row,mapped))/rows[i] for k in range(2)] if rows[i] else None
        means.append(mean);bary_atoms.append({'source_index':i,'point':mean,'mass':rows[i]})
    def moments(atoms):
        mean=[math.fsum(a['mass']*a['point'][k] for a in atoms if a['mass'])/total for k in range(2)]
        covariance=[[math.fsum(a['mass']*(a['point'][k]-mean[k])*(a['point'][l]-mean[l]) for a in atoms if a['mass'])/total for l in range(2)] for k in range(2)]
        return {'mean':mean,'covariance':covariance,'variance_trace':sum(covariance[k][k] for k in range(2))}
    full=moments(atoms);bary=moments(bary_atoms)
    conditional=[[math.fsum(a['mass']*(a['point'][k]-means[a['source_index']][k])*(a['point'][l]-means[a['source_index']][l]) for a in atoms if a['mass'])/total for l in range(2)] for k in range(2)]
    residual=max(abs(full['covariance'][k][l]-bary['covariance'][k][l]-conditional[k][l]) for k in range(2) for l in range(2))
    return {'time':t,'atoms':atoms,'barycentric_atoms':bary_atoms,'total_mass':total,
            'actual_source_mass':rows,'actual_target_mass':cols,'atom_count':len(atoms),
            'full_moments':full,'barycentric_moments':bary,
            'mean_within_source_covariance':conditional,'variance_decomposition_max_abs_residual':residual,
            'semantics':'unchanged joint atom masses; moments normalized by actual total; not a feasibility certificate, fitted model or sampled trajectory'}
