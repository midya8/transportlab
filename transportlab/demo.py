from .core import squared_cost,sinkhorn,exact_uniform_assignment,barycentric


def demo():
    x=[[-1.,-1.],[-1.,1.],[0.,0.],[1.,-1.],[1.,1.],[2.,0.]]
    perm=[3,0,5,2,1,4];y=[[x[i][0]+.6,x[i][1]-.3] for i in perm]
    c=squared_cost(x,y);w=[1/6]*6;exact=exact_uniform_assignment(c)
    arms=[]
    for eps in [.1,.5,2.]:
        r=sinkhorn(c,w,w,eps);arms.append({'epsilon':eps,'result':r,'barycentric':barycentric(r['plan'],y)})
        if eps!=.1:assert r['converged']
        else:assert not r['converged']
        # Cost comparison is only meaningful as a feasible approximation when converged.
        if r['converged']:assert r['transport_cost']>=exact['transport_cost']-1e-7
    independent=[[a*b for b in w] for a in w]
    independent_cost=sum(p*z for row,cr in zip(independent,c) for p,z in zip(row,cr))
    # An intentionally capped nonconverged result stays visible, never repaired.
    failed=sinkhorn([[0.,4.],[4.,0.]],[.9,.1],[.5,.5],.01,1)
    assert not failed['converged'] and failed['row_max_abs_residual']>.1
    self_result=sinkhorn(squared_cost(x,x),w,w,2.)
    assert self_result['transport_cost']>.1
    return {'fixture':'six synthetic2D points and a permuted translation(+.6,-.3); no learned representation or hidden correspondence evaluation',
            'source_points':x,'target_points':y,'cost_matrix':c,'weights':w,'arms':arms,
            'exact_uniform_assignment':exact,'independence':{'plan':independent,'transport_cost':independent_cost},
            'nonconverged_example':failed,'self_transport_epsilon2':self_result,
            'limits':'squared Euclidean cost is not a metric; entropic transport cost is biased and self-cost need not be zero; not a Sinkhorn divergence'}
