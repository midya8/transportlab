import copy
import itertools
import math
import unittest
from transportlab.core import sinkhorn,squared_cost,barycentric,exact_uniform_assignment


class TransportTests(unittest.TestCase):
    def test_two_by_two_closed_form(self):
        # Symmetric uniform case: diagonal mass .5/(1+exp(-d/eps)).
        for eps in [.01,.5,2.]:
            r=sinkhorn([[0.,2.],[2.,0.]],[.5,.5],[.5,.5],eps)
            diagonal=.5/(1+math.exp(-2/eps))
            self.assertTrue(r['converged']);self.assertAlmostEqual(r['plan'][0][0],diagonal)
            self.assertAlmostEqual(r['plan'][0][1],.5-diagonal)
            self.assertAlmostEqual(r['transport_cost'],2*(1-2*diagonal))

    def test_zero_cost_is_independence(self):
        a,b=[.2,.3,.5],[.4,.6];r=sinkhorn([[0.,0.]]*3,a,b)
        for i in range(3):
            for j in range(2):self.assertAlmostEqual(r['plan'][i][j],a[i]*b[j])
        self.assertAlmostEqual(r['transport_cost'],0.)
        expected=-sum(x*math.log(x) for x in a)-sum(x*math.log(x) for x in b)
        self.assertAlmostEqual(r['entropy_nats'],expected)
        self.assertAlmostEqual(r['regularized_objective'],-.5*expected)

    def test_zero_weights_preserved(self):
        r=sinkhorn([[0.,2.,1.],[3.,4.,5.]],[0.,1.],[.25,0.,.75])
        self.assertEqual(r['plan'][0],[0.,0.,0.]);self.assertEqual(r['plan'][1][1],0.)
        self.assertAlmostEqual(r['plan'][1][0],.25);self.assertAlmostEqual(r['plan'][1][2],.75)
        self.assertEqual(r['active_shape'],[1,2])

    def test_direct_primal_scaling_reference(self):
        cost=[[.2,1.,.5],[1.3,.1,.8]];a,b=[.3,.7],[.2,.5,.3];eps=.7
        k=[[math.exp(-z/eps) for z in row] for row in cost];u=[1.,1.];v=[1.,1.,1.]
        for _ in range(1000):
            u=[a[i]/sum(k[i][j]*v[j] for j in range(3)) for i in range(2)]
            v=[b[j]/sum(k[i][j]*u[i] for i in range(2)) for j in range(3)]
        r=sinkhorn(cost,a,b,eps,tolerance=1e-12)
        self.assertTrue(r['converged'])
        for i in range(2):
            for j in range(3):self.assertAlmostEqual(r['plan'][i][j],u[i]*k[i][j]*v[j],places=11)

    def test_kernel_underflow_does_not_destroy_log_solution(self):
        self.assertEqual(math.exp(-1000/.001),0.)
        r=sinkhorn([[1000.,1002.],[1002.,1000.]],[.5,.5],[.5,.5],.001)
        self.assertTrue(r['converged']);self.assertEqual(r['underflowed_active_entries'],2)
        self.assertAlmostEqual(r['plan'][0][0],.5,places=9)
        self.assertAlmostEqual(r['transport_cost'],1000.,places=6)

    def test_common_cost_shift_and_scale_equivariance(self):
        c=[[.1,1.],[2.,.3]];a,b=[.4,.6],[.7,.3]
        r=sinkhorn(c,a,b,.7,tolerance=1e-12)
        shifted=sinkhorn([[z+1000 for z in row] for row in c],a,b,.7,tolerance=1e-12)
        scaled=sinkhorn([[z*3 for z in row] for row in c],a,b,2.1,tolerance=1e-12)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(r['plan'][i][j],shifted['plan'][i][j],places=11)
                self.assertAlmostEqual(r['plan'][i][j],scaled['plan'][i][j],places=11)

    def test_rectangular_marginals_and_permutation(self):
        c=[[0.,2.,1.],[1.,0.,3.]];a,b=[.3,.7],[.2,.5,.3]
        r=sinkhorn(c,a,b,tolerance=1e-12)
        self.assertTrue(r['converged']);self.assertLess(r['row_max_abs_residual'],1e-12)
        rr=sinkhorn([list(reversed(row)) for row in reversed(c)],list(reversed(a)),list(reversed(b)),tolerance=1e-12)
        for i in range(2):
            for j in range(3):self.assertAlmostEqual(r['plan'][i][j],rr['plan'][1-i][2-j])

    def test_nonconvergence_is_not_repaired(self):
        r=sinkhorn([[0.,4.],[4.,0.]],[.9,.1],[.5,.5],.01,1)
        self.assertFalse(r['converged']);self.assertEqual(r['iterations'],1)
        self.assertGreater(r['row_max_abs_residual'],.39)
        self.assertLess(r['column_max_abs_residual'],1e-12)
        self.assertAlmostEqual(sum(r['plan'][0]),.5)

    def test_finite_epsilon_self_cost_is_not_zero(self):
        r=sinkhorn([[0.,1.],[1.,0.]],[.5,.5],[.5,.5],1.)
        self.assertGreater(r['transport_cost'],.2)

    def test_assignment_enumeration_ties_and_scope(self):
        r=exact_uniform_assignment([[4.,1.,3.],[2.,0.,5.],[3.,2.,2.]])
        self.assertEqual(r['permutation'],[1,0,2]);self.assertAlmostEqual(r['transport_cost'],5/3)
        self.assertEqual(r['permutations_checked'],6)
        self.assertEqual(exact_uniform_assignment([[0.,0.],[0.,0.]])['permutation'],[0,1])
        with self.assertRaises(ValueError):exact_uniform_assignment([[0.]*9 for _ in range(9)])

    def test_assignment_minimum_matches_unregularized_2x2_endpoints(self):
        for c in [[[0.,3.],[2.,0.]],[[4.,1.],[0.,3.]]]:
            reference=exact_uniform_assignment(c)['transport_cost']
            # Every 2x2 feasible uniform plan is parameterized by t in [0,.5].
            vals=[t*c[0][0]+(.5-t)*c[0][1]+(.5-t)*c[1][0]+t*c[1][1] for t in [0.,.5]]
            self.assertEqual(reference,min(vals))

    def test_squared_cost_and_barycentric_is_mean_not_assignment(self):
        y=[[0.,0.],[2.,2.]];self.assertEqual(squared_cost([[1.,1.]],y),[[2.,2.]])
        r=barycentric([[.25,.25],[0.,0.]],y)
        self.assertEqual(r['points'],[[1.,1.],None]);self.assertEqual(r['actual_row_mass'],[.5,0.])
        self.assertNotIn(r['points'][0],y)

    def test_no_input_mutation_and_reproducible(self):
        c=[[0.,1.],[1.,0.]];a=[.3,.7];b=[.5,.5];old=copy.deepcopy((c,a,b))
        self.assertEqual(sinkhorn(c,a,b),sinkhorn(c,a,b));self.assertEqual((c,a,b),old)

    def test_invalid_inputs(self):
        for c,a,b in [([[0.]],[1.],[0.]),([[0.]],[True],[1.]),([[0.]],[.5],[1.]),
                      ([[float('nan')]],[1.],[1.]),([[-1.]],[1.],[1.]),([[0.,1.]],[1.],[1.]),
                      ([[0.]],[1.,0.],[1.]),([[0.]],[float('inf')],[1.])]:
            with self.assertRaises(ValueError):sinkhorn(c,a,b)
        for kw in [{'epsilon':0},{'epsilon':True},{'tolerance':0},{'max_iterations':True},{'max_iterations':5001}]:
            with self.assertRaises(ValueError):sinkhorn([[0.]],[1.],[1.],**kw)
        with self.assertRaises(ValueError):sinkhorn([[0.]*64 for _ in range(64)],[1/64]*64,[1/64]*64)
        for x in [[[True,0.]],[[101.,0.]],[[1.]],[]]:
            with self.assertRaises(ValueError):squared_cost(x,[[0.,0.]])
        with self.assertRaises(ValueError):barycentric([[-1.,2.]],[[0.,0.],[1.,1.]])
