from .audit import audit_transport
from .sensitivity import differentiate_iterations
from .core import squared_cost,sinkhorn,exact_uniform_assignment,barycentric,pushforward


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
    push=[]
    for arm in arms:
        if not arm['result']['converged']:continue
        slices=[pushforward(arm['result']['plan'],x,y,t) for t in [0.,.5,1.]]
        assert max(s['variance_decomposition_max_abs_residual'] for s in slices)<1e-12
        push.append({'epsilon':arm['epsilon'],'slices':slices})
    objective_audits=[{'epsilon':e,'audit':audit_transport(c,w,w,e)} for e in [.1,.5,2.]]
    objective_audits.append({'fixture':'one-iteration unequal marginal failure','audit':audit_transport([[0.,4.],[4.,0.]],[.9,.1],[.5,.5],.01,1)})
    assert all(z['audit']['rounded']['row_max_abs_residual']<1e-12 and z['audit']['rounded']['column_max_abs_residual']<1e-12 and z['audit']['signed_numerical_gap']>=-1e-8 for z in objective_audits)
    sensitivities=[]
    sc=[[.2,3.],[1.,.4]];sd=[[.1,.8],[-.4,.3]];sa=[.9,.1];sb=[.5,.5]
    for steps in [1,10,100]:
        r=differentiate_iterations(sc,sa,sb,sd,.5,.2,steps);checks=[]
        for h in [1e-4,1e-5,1e-6]:
            sides=[differentiate_iterations([[sc[i][j]+sign*h*sd[i][j] for j in range(2)] for i in range(2)],sa,sb,[[0.,0.],[0.,0.]],.5+sign*h*.2,0.,steps) for sign in [-1,1]]
            fd=(sides[1]['regularized_objective']-sides[0]['regularized_objective'])/(2*h)
            pe=max(abs((sides[1]['plan'][i][j]-sides[0]['plan'][i][j])/(2*h)-r['plan_directional_derivative'][i][j]) for i in range(2) for j in range(2))
            checks.append({'step_size':h,'objective_central_difference':fd,'objective_abs_error':abs(fd-r['regularized_objective_directional_derivative']),'plan_max_abs_error':pe})
        assert max(z['objective_abs_error'] for z in checks)<1e-7 and r['chain_identity_abs_residual']<1e-12
        sensitivities.append({'result':r,'finite_difference_checks':checks})
    assert abs(sensitivities[0]['result']['regularized_objective_directional_derivative']-sensitivities[0]['result']['envelope_shortcut'])>1e-3
    return {'sensitivity_audit':{'cost':sc,'cost_direction':sd,'source_weights':sa,'target_weights':sb,'epsilon_direction':.2,'arms':sensitivities},'objective_audits':objective_audits,'pushforward_audit':push,'fixture':'six synthetic2D points and a permuted translation(+.6,-.3); no learned representation or hidden correspondence evaluation',
            'source_points':x,'target_points':y,'cost_matrix':c,'weights':w,'arms':arms,
            'exact_uniform_assignment':exact,'independence':{'plan':independent,'transport_cost':independent_cost},
            'nonconverged_example':failed,'self_transport_epsilon2':self_result,
            'limits':'squared Euclidean cost is not a metric; entropic transport cost is biased and self-cost need not be zero; not a Sinkhorn divergence'}
