# File to test the performance of the f function on the
# classical Newtons method. (Task 5)
import numpy as np
import unittest as ut
#from ClassicalNewton import *
from ClassicalNewton import *
from OptimizationProblem import *
def f(alpha):
    return 100*(alpha**4) + (1-alpha)**2

def df(alpha):
    return 400*(alpha**3) - 2*(1-alpha)

grad = [df]

class separateTestCase(ut.TestCase):
    def setUp(self):
        expected = 0.1609
    #Test Goldstein
    def testGoldsteinConditionsWithSuppliedGradientOne(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        actual = op.solve((0.1))
        self.assert_almost_equal(expected, actual, 0.0001)
    def testGoldsteinConditionsWithSuppliedGradientTwo(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        actual = op.solve((1))
        self.assert_almost_equal(expected, actual, 0.0001)
    def testGoldsteinConditionsWithoutSuppliedGradientOne(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        actual = op.solve((0.1))
        self.assert_almost_equal(expected, actual, 0.0001)
    def testGoldsteinConditionsWithSuppliedGradientTwo(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        actual = op.solve((1))
        self.assert_almost_equal(expected, actual, 0.0001)

    #Test Wolfstein
    """
    def testWolfePowellConditionsWithSuppliedGradientOne(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(Problem, ws)
        actual = op.solve(0.1)
        self.assert_almost_equal(expected, actual, 0.0001)
    def testWolfePowellConditionsWithSuppliedGradientTwo(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(Problem)
        actual = op.solve(1)
        self.assert_almost_equal(expected, actual, 0.0001)
    def testWolfePowellConditionsWithoutSuppliedGradientOne(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(Problem, ws)
        actual = op.solve(0.1)
        self.assert_almost_equal(expected, actual, 0.0001)
    def testWolfePowellConditionsWithoutSuppliedGradientTwo(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(Problem)
        actual = op.solve(1)
        self.assert_almost_equal(expected, actual, 0.0001)
    """
if __name__=='__main__':
    ut.main()
