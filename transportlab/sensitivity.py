"""Forward directional derivatives of a fixed Sinkhorn computation, not an optimum oracle."""
import math
from .core import weights,matrix,number,integer


def differentiate_iterations(cost,source,target,cost_direction,epsilon=.5,epsilon_direction=0.,iterations=100):
    a,_=weights(source,'source');b,_=weights(target,'target');c=matrix(cost,len(a),len(b))
    if not isinstance(cost_direction,list) or len(cost_direction)!=len(a):raise ValueError('cost direction row mismatch')
    dc=[]
    for row in cost_direction:
        if not isinstance(row,list) or len(row)!=len(b):raise ValueError('cost direction column mismatch')
        r=[number(z,'cost direction') for z in row]
        if any(abs(z)>1e6 for z in r):raise ValueError('cost direction outside[-1e6,1e6]')
        dc.append(r)
    eps=number(epsilon,'epsilon');deps=number(epsilon_direction,'epsilon direction')
    if not 1e-4<=eps<=1e4:raise ValueError('epsilon outside[1e-4,1e4]')
    if abs(deps)>1e4:raise ValueError('epsilon direction outside[-1e4,1e4]')
    steps=integer(iterations,1,5000,'iterations')
    rows=[i for i,z in enumerate(a) if z];cols=[j for j,z in enumerate(b) if z]
    if len(rows)*len(cols)*steps>10000000:raise ValueError('active cells times iterations exceeds10000000')
    k=[[-c[i][j]/eps for j in cols] for i in rows]
    dk=[[-dc[i][j]/eps+c[i][j]*deps/(eps*eps) for j in cols] for i in rows]
    u=[0.]*len(rows);v=[0.]*len(cols);du=u[:];dv=v[:]
    def lse_tangent(z,dz):
        high=max(z);e=[math.exp(x-high) for x in z];total=math.fsum(e)
        return high+math.log(total),math.fsum(x*d for x,d in zip(e,dz))/total
    for _ in range(steps):
        for i,ii in enumerate(rows):
            value,deriv=lse_tangent([k[i][j]+v[j] for j in range(len(cols))],[dk[i][j]+dv[j] for j in range(len(cols))])
            u[i]=math.log(a[ii])-value;du[i]=-deriv
        for j,jj in enumerate(cols):
            value,deriv=lse_tangent([k[i][j]+u[i] for i in range(len(rows))],[dk[i][j]+du[i] for i in range(len(rows))])
            v[j]=math.log(b[jj])-value;dv[j]=-deriv
        shift=math.fsum(u)/len(u);dshift=math.fsum(du)/len(du)
        u=[x-shift for x in u];du=[x-dshift for x in du];v=[x+shift for x in v];dv=[x+dshift for x in dv]
    p=[[0.]*len(b) for _ in a];dp=[[0.]*len(b) for _ in a];logs=[];dlogs=[];active=[]
    for i,ii in enumerate(rows):
        for j,jj in enumerate(cols):
            lp=u[i]+k[i][j]+v[j];dlp=du[i]+dk[i][j]+dv[j];mass=math.exp(lp);d=mass*dlp
            p[ii][jj]=mass;dp[ii][jj]=d;logs.append(lp);dlogs.append(dlp);active.append((ii,jj))
    rr=[math.fsum(row) for row in p];cc=[math.fsum(row[j] for row in p) for j in range(len(b))]
    dr=[math.fsum(row) for row in dp];dd=[math.fsum(row[j] for row in dp) for j in range(len(b))]
    transport=math.fsum(p[i][j]*c[i][j] for i,j in active)
    dtransport=math.fsum(dp[i][j]*c[i][j]+p[i][j]*dc[i][j] for i,j in active)
    sum_plogp=math.fsum(p[i][j]*lp for (i,j),lp in zip(active,logs))
    dsum_plogp=math.fsum(dp[i][j]*(lp+1) for (i,j),lp in zip(active,logs))
    obj=transport+eps*sum_plogp;dobj=dtransport+deps*sum_plogp+eps*dsum_plogp
    envelope=math.fsum(p[i][j]*dc[i][j] for i,j in active)+deps*sum_plogp
    # At scaled P, C+eps*(log(P)+1) = f_i+g_j. Keep residual chain term explicit.
    chain=math.fsum((eps*u[i])*dr[ii] for i,ii in enumerate(rows))+math.fsum(eps*(v[j]+1)*dd[jj] for j,jj in enumerate(cols))
    values=[z for row in dp for z in row]+[transport,dtransport,obj,dobj,envelope,chain]
    if not all(math.isfinite(z) for z in values):raise ValueError('nonfinite sensitivity result')
    return {'plan':p,'plan_directional_derivative':dp,'iterations':steps,'epsilon':eps,'epsilon_direction':deps,
            'transport_cost':transport,'transport_cost_directional_derivative':dtransport,
            'regularized_objective':obj,'regularized_objective_directional_derivative':dobj,
            'envelope_shortcut':envelope,'residual_chain_term':chain,
            'chain_identity_abs_residual':abs(dobj-envelope-chain),
            'actual_source_mass':rr,'actual_target_mass':cc,'source_mass_directional_derivative':dr,'target_mass_directional_derivative':dd,
            'row_max_abs_residual':max(abs(x-y) for x,y in zip(rr,a)),
            'column_max_abs_residual':max(abs(x-y) for x,y in zip(cc,b)),
            'underflowed_active_entries':sum(p[i][j]==0 for i,j in active),
            'semantics':'analytic forward directional derivative through exactly iterations updates from zero log scalings; fixed marginals; no adaptive stopping, implicit optimum, rounded-plan derivative or certified real-arithmetic derivative'}
