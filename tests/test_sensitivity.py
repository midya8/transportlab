import copy
import json
import math
import subprocess
import sys
import unittest
from transportlab.sensitivity import differentiate_iterations as run

class SensitivityTests(unittest.TestCase):
    def test_independent_central_differences(self):
        c=[[.4,3.,2.],[4.,1.,.2]];d=[[.3,-.4,.5],[.2,.6,-.8]];a=[.6,.4];b=[.1,.3,.6]
        for n in [1,3,100]:
            for de in [0.,.2]:
                r=run(c,a,b,d,.7,de,n);h=1e-5
                q=[run([[x+s*h*y for x,y in zip(row,di)] for row,di in zip(c,d)],a,b,[[0.]*3 for _ in a],.7+s*h*de,0.,n) for s in [-1,1]]
                for i in range(2):
                    for j in range(3):self.assertAlmostEqual(r['plan_directional_derivative'][i][j],(q[1]['plan'][i][j]-q[0]['plan'][i][j])/(2*h),8)
                for key in ['transport_cost','regularized_objective']:self.assertAlmostEqual(r[key+'_directional_derivative'],(q[1][key]-q[0][key])/(2*h),8)
                self.assertLess(r['chain_identity_abs_residual'],1e-12)
    def test_closed_two_by_two(self):
        eps=.7;r=run([[0.,1.],[1.,0.]],[.5,.5],[.5,.5],[[0.,1.],[1.,0.]],eps,0.,10)
        q=math.exp(-1/eps);derivative=-q/(2*eps*(1+q)**2)
        self.assertAlmostEqual(r['plan_directional_derivative'][0][1],derivative,13)
        self.assertAlmostEqual(r['regularized_objective_directional_derivative'],q/(1+q),13)
    def test_cost_shift(self):
        r=run([[.2,3.],[1.,.4]],[.7,.3],[.4,.6],[[1.,1.],[1.,1.]],.5,0.,5)
        self.assertLess(max(abs(x) for row in r['plan_directional_derivative'] for x in row),1e-14)
        self.assertAlmostEqual(r['regularized_objective_directional_derivative'],1.,12)
    def test_simultaneous_scale(self):
        c=[[.2,3.],[1.,.4]];r=run(c,[.7,.3],[.4,.6],c,.5,.5,3)
        self.assertEqual(r['plan_directional_derivative'],[[0.,0.],[0.,0.]])
        self.assertAlmostEqual(r['regularized_objective_directional_derivative'],r['regularized_objective'],13)
    def test_nonconverged_envelope_counterexample(self):
        r=run([[.2,3.],[1.,.4]],[.9,.1],[.5,.5],[[.1,.8],[-.4,.3]],.5,.2,1)
        self.assertGreater(r['row_max_abs_residual'],.1)
        self.assertGreater(abs(r['regularized_objective_directional_derivative']-r['envelope_shortcut']),1e-3)
        self.assertLess(r['chain_identity_abs_residual'],1e-12)
        self.assertLess(max(abs(x) for x in r['target_mass_directional_derivative']),1e-14)
    def test_zero_support_underflow_and_no_mutation(self):
        c=[[0.,1e6],[1e6,0.]];d=[[1.,2.],[3.,4.]];old=copy.deepcopy((c,d))
        r=run(c,[.5,.5],[.5,.5],d,1e-4,0.,3);self.assertEqual(r['underflowed_active_entries'],2);self.assertEqual((c,d),old)
        r=run(c,[0.,1.],[1.,0.],d,1e-4,0.,3)
        self.assertEqual(r['plan_directional_derivative'],[[0.,0.],[0.,0.]])
        self.assertAlmostEqual(r['regularized_objective_directional_derivative'],3.,5)
    def test_permutation(self):
        c=[[.2,3.],[1.,.4]];d=[[.1,.8],[-.4,.3]];r=run(c,[.9,.1],[.5,.5],d,.5,.2,3)
        q=run([c[1][::-1],c[0][::-1]],[.1,.9],[.5,.5],[d[1][::-1],d[0][::-1]],.5,.2,3)
        self.assertAlmostEqual(r['regularized_objective_directional_derivative'],q['regularized_objective_directional_derivative'],12)
        for i in range(2):
            for j in range(2):self.assertAlmostEqual(r['plan_directional_derivative'][i][j],q['plan_directional_derivative'][1-i][1-j],12)
    def test_invalid(self):
        for d in [[],[[float('nan')]],[[1e7]],[[True]],[[0.,1.]]]:
            with self.assertRaises(ValueError):run([[1.]],[1.],[1.],d)
        for n in [0,5001,True]:
            with self.assertRaises(ValueError):run([[1.]],[1.],[1.],[[0.]],iterations=n)
        with self.assertRaises(ValueError):run([[0.]*64 for _ in range(64)],[1/64]*64,[1/64]*64,[[0.]*64 for _ in range(64)],iterations=5000)
    def test_primal_equals_independent_solver_at_fixed_cap(self):
        from transportlab.core import sinkhorn
        c=[[.2,3.],[1.,.4]];a=[.9,.1];b=[.5,.5]
        for n in [1,3,10]:
            r=run(c,a,b,[[0.,0.],[0.,0.]],iterations=n);q=sinkhorn(c,a,b,max_iterations=n,tolerance=1e-12)
            self.assertEqual(q['iterations'],n)
            for i in range(2):
                for j in range(2):self.assertAlmostEqual(r['plan'][i][j],q['plan'][i][j],14)
            self.assertAlmostEqual(r['regularized_objective'],q['regularized_objective'],14)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','transportlab','sensitivity','examples/direction.json','--iterations','1'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertEqual(v['iterations'],1);self.assertIn('residual_chain_term',v)
        for extra in [['--iterations','0'],['--epsilon','nan']]:
            q=subprocess.run([sys.executable,'-m','transportlab','sensitivity','examples/direction.json']+extra,capture_output=True,text=True);self.assertEqual(q.returncode,2);self.assertNotIn('Traceback',q.stderr)
