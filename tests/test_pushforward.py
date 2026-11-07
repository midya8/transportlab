import copy
import math
import unittest
from transportlab.core import pushforward


class PushforwardTests(unittest.TestCase):
    def test_independent_map_mean_collapses_but_joint_atoms_do_not(self):
        p=[[.25,.25],[.25,.25]];x=[[-1.,0.],[1.,0.]];y=[[-2.,0.],[2.,0.]]
        r=pushforward(p,x,y,1.)
        self.assertEqual(r['atom_count'],4);self.assertEqual(r['barycentric_moments']['variance_trace'],0.)
        self.assertEqual(r['full_moments']['variance_trace'],4.)
        self.assertEqual(r['mean_within_source_covariance'],[[4.,0.],[0.,0.]])
        self.assertEqual(r['variance_decomposition_max_abs_residual'],0.)
        self.assertEqual([a['mass'] for a in r['atoms']],[.25]*4)

    def test_endpoint_marginals_and_midpoint_mean(self):
        p=[[.1,.2],[.3,.4]];x=[[-1.,2.],[3.,-2.]];y=[[0.,1.],[4.,5.]]
        a,b,m=[pushforward(p,x,y,t) for t in [0.,1.,.5]]
        self.assertAlmostEqual(a['actual_source_mass'][0],.3)
        self.assertAlmostEqual(a['actual_target_mass'][0],.4)
        for atom in a['atoms']:self.assertEqual(atom['point'],x[atom['source_index']])
        for atom in b['atoms']:self.assertEqual(atom['point'],y[atom['target_index']])
        for k in range(2):self.assertAlmostEqual(m['full_moments']['mean'][k],(a['full_moments']['mean'][k]+b['full_moments']['mean'][k])/2)

    def test_deterministic_plan_has_no_conditional_variance(self):
        r=pushforward([[.5,0.],[0.,.5]],[[0.,0.],[2.,2.]],[[1.,1.],[3.,3.]],.7)
        self.assertEqual(r['full_moments'],r['barycentric_moments'])
        self.assertEqual(r['mean_within_source_covariance'],[[0.,0.],[0.,0.]])
        self.assertEqual(len(r['atoms']),4)  # Zero cells are retained, not thresholded away.

    def test_law_total_covariance_and_t_squared_gap(self):
        p=[[.1,.2,.1],[.2,.1,.3]];x=[[-1.,2.],[3.,-2.]];y=[[0.,1.],[4.,5.],[-2.,1.]]
        a=pushforward(p,x,y,1.);b=pushforward(p,x,y,.3)
        for r in [a,b]:
            self.assertLess(r['variance_decomposition_max_abs_residual'],1e-14)
            for k in range(2):self.assertAlmostEqual(r['full_moments']['mean'][k],r['barycentric_moments']['mean'][k])
        for k in range(2):
            for l in range(2):self.assertAlmostEqual(b['mean_within_source_covariance'][k][l],.09*a['mean_within_source_covariance'][k][l])
        c=b['mean_within_source_covariance'];self.assertGreaterEqual(c[0][0],0.);self.assertGreaterEqual(c[1][1],0.)
        self.assertGreaterEqual(c[0][0]*c[1][1]-c[0][1]*c[1][0],-1e-14)

    def test_mass_scale_changes_atoms_not_normalized_moments(self):
        p=[[.1,.2],[.3,.4]];x=[[0.,0.],[2.,2.]];y=[[-1.,1.],[1.,-1.]]
        a=pushforward(p,x,y);b=pushforward([[v*5 for v in row] for row in p],x,y)
        self.assertAlmostEqual(b['total_mass'],5*a['total_mass'])
        for k in range(2):self.assertAlmostEqual(a['full_moments']['mean'][k],b['full_moments']['mean'][k])
        self.assertEqual(b['atoms'][0]['mass'],.5)

    def test_zero_rows_and_no_mutation(self):
        p=[[0.,0.],[.3,.7]];x=[[1.,1.],[2.,2.]];y=[[0.,0.],[4.,4.]];old=copy.deepcopy((p,x,y))
        r=pushforward(p,x,y);self.assertEqual(r['barycentric_atoms'][0],{'source_index':0,'point':None,'mass':0.})
        self.assertEqual((p,x,y),old)

    def test_invalid_plan_and_time(self):
        x=[[0.,0.]];y=[[1.,1.]]
        for p in [[],[[0.]],[[True]],[[-1.]],[[float('inf')]],[[1.,0.]],[[1e7]]]:
            with self.assertRaises(ValueError):pushforward(p,x,y)
        for t in [True,-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):pushforward([[1.]],x,y,t)

    def test_push_cli_convergence_gate_and_time(self):
        import json,subprocess,sys,tempfile,os
        with tempfile.TemporaryDirectory() as d:
            f=os.path.join(d,'input.json')
            v={'source_points':[[0.,0.],[1.,1.]],'target_points':[[.2,.2],[1.2,1.2]],'source_weights':[.5,.5],'target_weights':[.5,.5]}
            with open(f,'w') as h:json.dump(v,h)
            p=subprocess.run([sys.executable,'-m','transportlab','push',f,'--time','.5'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);r=json.loads(p.stdout)
            self.assertTrue(r['solver']['converged']);self.assertEqual(r['pushforward']['atom_count'],4)
            p=subprocess.run([sys.executable,'-m','transportlab','push',f,'--time','2'],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
        p=subprocess.run([sys.executable,'-m','transportlab','push','examples/points.json','--epsilon','.1'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertIn('requires a converged plan',p.stderr)
