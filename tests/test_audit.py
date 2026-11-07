import copy
import json
import math
import subprocess
import sys
import unittest
from transportlab.audit import round_plan,audit_transport
from transportlab.core import sinkhorn

class AuditTests(unittest.TestCase):
    def feasible(self,r,a,b):
        self.assertLess(r['row_max_abs_residual'],1e-12);self.assertLess(r['column_max_abs_residual'],1e-12)
        self.assertTrue(all(v>=0 for row in r['plan'] for v in row))
    def test_round_empty_and_rectangular(self):
        for p in [[[0.,0.,0.],[0.,0.,0.]],[[10.,2.,0.],[0.,0.,9.]]]:
            q=copy.deepcopy(p);r=round_plan(p,[.7,.3],[.1,.4,.5]);self.feasible(r,[.7,.3],[.1,.4,.5]);self.assertEqual(p,q)
        self.assertEqual(round_plan([[0.,0.],[0.,0.]],[.5,.5],[.25,.75])['plan'],[[.125,.375],[.125,.375]])
    def test_feasible_unchanged(self):
        p=[[.125,.375],[.125,.375]];r=round_plan(p,[.5,.5],[.25,.75]);self.assertEqual(p,r['plan']);self.assertEqual(r['l1_plan_change'],0.)
    def test_closed_two_by_two(self):
        eps=.7;r=audit_transport([[0.,1.],[1.,0.]],[.5,.5],[.5,.5],eps)
        exact=-eps*math.log(2*(1+math.exp(-1/eps)))
        self.assertAlmostEqual(r['dual_value'],exact,12);self.assertAlmostEqual(r['rounded']['regularized_objective'],exact,12);self.assertAlmostEqual(r['signed_numerical_gap'],0.,12)
        self.assertGreater(r['rounded']['transport_cost'],0.) # not an unregularized zero-cost certificate
    def test_failed_raw_objective_is_not_primal_bound(self):
        c=[[0.,4.],[4.,0.]];a=[.9,.1];b=[.5,.5];r=audit_transport(c,a,b,.01,1)
        self.assertFalse(r['raw_solver']['converged']);self.assertGreater(r['raw_solver']['row_max_abs_residual'],.39)
        self.assertLess(r['raw_objective_minus_dual'],0.)
        self.feasible(r['rounded'],a,b);self.assertGreater(r['signed_numerical_gap'],1.5)
        self.assertGreater(r['rounded']['l1_plan_change'],.79)
    def test_potentials_reconstruct_and_gauge(self):
        c=[[0.,3.,2.],[4.,1.,0.]];a=[.6,.4];b=[.1,.3,.6];r=audit_transport(c,a,b,.5)
        f,g=r['dual_source_potential'],r['dual_target_potential']
        for i in range(2):
            for j in range(3):self.assertAlmostEqual(math.exp((f[i]+g[j]-c[i][j])/.5-1),r['raw_solver']['plan'][i][j],13)
        shift=3.
        d=sum(a[i]*(f[i]+shift) for i in range(2))+sum(b[j]*(g[j]-shift) for j in range(3))-.5*r['dual_exponential_mass']
        self.assertAlmostEqual(d,r['dual_value'],13);self.assertGreaterEqual(r['signed_numerical_gap'],-1e-12)
    def test_zero_marginals_and_underflow(self):
        r=audit_transport([[0.,1.],[1e6,0.]],[0.,1.],[1.,0.],1e-4)
        self.assertIsNone(r['dual_source_potential'][0]);self.assertIsNone(r['dual_target_potential'][1]);self.feasible(r['rounded'],[0.,1.],[1.,0.])
        r=audit_transport([[0.,1e6],[1e6,0.]],[.5,.5],[.5,.5],1e-4)
        self.assertEqual(r['raw_solver']['underflowed_active_entries'],2);self.assertLess(abs(r['signed_numerical_gap']),1e-8)
    def test_scaling_opt_in_and_cost_shift(self):
        c=[[0.,1.],[1.,0.]];a=[.5,.5]
        self.assertNotIn('log_scaling',sinkhorn(c,a,a))
        r=audit_transport(c,a,a);q=audit_transport([[x+2 for x in row] for row in c],a,a)
        self.assertAlmostEqual(q['dual_value']-r['dual_value'],2.,12);self.assertAlmostEqual(q['signed_numerical_gap'],r['signed_numerical_gap'],12)
    def test_invalid(self):
        for p in [[[float('nan')]], [[-1.]], [[1000001.]], [], [[]]]:
            with self.assertRaises(ValueError):round_plan(p,[1.],[1.])
        with self.assertRaises(ValueError):sinkhorn([[0.]],[1.],[1.],include_scaling=1)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','transportlab','audit','examples/points.json','--epsilon','.1'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertFalse(v['raw_solver']['converged']);self.assertIn('signed_numerical_gap',v)
        q=subprocess.run([sys.executable,'-m','transportlab','audit','examples/points.json','--epsilon','0'],capture_output=True,text=True);self.assertEqual(q.returncode,2);self.assertNotIn('Traceback',q.stderr)
    def test_zero_cost_independent_optimum(self):
        a=[.2,.3,.5];b=[.6,.4];r=audit_transport([[0.,0.] for _ in a],a,b,.8)
        entropy=-sum(v*math.log(v) for v in a+b)
        self.assertAlmostEqual(r['dual_value'],-.8*entropy,12)
        self.assertLess(abs(r['signed_numerical_gap']),1e-12)
        for i in range(3):
            for j in range(2):self.assertAlmostEqual(r['rounded']['plan'][i][j],a[i]*b[j],13)
    def test_weak_duality_independent_feasible_plans(self):
        # Enumerate a one-dimensional 2x2 transport polytope, not solver-derived primals.
        c=[[1.,4.],[2.,.3]];a=[.7,.3];b=[.4,.6];eps=.9;r=audit_transport(c,a,b,eps)
        for k in range(1001):
            z=.1+.3*k/1000;p=[[z,.7-z],[.4-z,z-.1]]
            primal=sum(p[i][j]*c[i][j]+eps*p[i][j]*math.log(p[i][j]) for i in range(2) for j in range(2) if p[i][j]>0)
            self.assertGreaterEqual(primal+1e-12,r['dual_value'])
    def test_deterministic_random_rounding_and_gaps(self):
        import random
        rng=random.Random(17)
        for _ in range(30):
            a=[rng.random() for _ in range(3)];b=[rng.random() for _ in range(4)];a=[x/sum(a) for x in a];b=[x/sum(b) for x in b]
            p=[[rng.random()*10 for _ in b] for _ in a];self.feasible(round_plan(p,a,b),a,b)
            c=[[rng.random()*5 for _ in b] for _ in a];r=audit_transport(c,a,b,.3,1)
            self.assertGreaterEqual(r['signed_numerical_gap'],-1e-10);self.feasible(r['rounded'],a,b)
