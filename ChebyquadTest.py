# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:08:23 2015

@author: Olle
"""
import scipy.optimize as so
from ClassicalNewton import *
from OptimizationProblem import *
from scipy import linspace
from chebyquad_problem_1 import *
import unittest as ut
class chebyquadTest(ut.TestCase):
    def setup(self):
        self.x = linspace(0,1,8)
        self.expected = so.fmin_bfgs(chebyquad,x,gradchebyquad)
    def testChebyquadWithExactLinesearchAndEstimatedGradient(self):
        def chebyquad_wrapper(*x):
            return chebyquad(x)
        problem = OptimizationProblem(chebyquad_wrapper)
       # gradient = gradchebyquad
        op = ClassicalNewton(problem)
        x = linspace(0,1,8)
        print(x)
        actual = op.solve(x)
        np.testing.assert_almost_equal(self.expected, actual)
        """
    def testChebyquadWithExactLinesearchAndSuppliedGradient(self):
        problem = OptimizationProblem(chebyquad)
        gradient = gradchebyquad
        op = ClassicalNewton(problem, gradient)
        actual = op.solve(search='exact')
        np.testing.assert_almost_equal(self.expected, actual)
    def testChebyquadWithInexactLinesearchAndSuppliedGradientAndGSCond(self):
        problem = OptimizationProblem(chebyquad)
        gradient = gradchebyquad
        op = ClassicalNewton(problem, gradient)
        actual = op.solve(search='inexact', cond='GS')
        np.testing.assert_almost_equal(self.expected, actual)
    def testChebyquadWithInexactLinesearchAndSuppliedGradientAndWPCond(self):
        problem = OptimizationProblem(chebyquad)
        gradient = gradchebyguad
        op = ClassicalNewton(problem, gradient)
        actual = op.solve(search='inexact', cond='GS')
        np.testing.assert_almost_equal(self.expected, actual)
    def testChebyquadWithInexactLinesearchAndEstimatedGradientAndGSCond(self):
        problem = OptimizationProblem(chebyquad)
        op = ClassicalNewton(problem)
        actual = op.solve(search='inexact', cond='GS')
        np.testing.assert_almost_equal(self.expected, actual)
    def testChebyquadWithInexactLinesearchAndEstimatedGradientAndWPCond(self):
        problem = OptimizationProblem(chebyquad)
        op = ClassicalNewton(problem)
        actual = op.solve(search='inexact', cond='WP')
        np.testing.assert_almost_equal(self.expected, actual)
        """
if __name__=='__main__':
    ut.main()
