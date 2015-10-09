# File to test the performance of the f function on the
# classical Newtons method. (Task 5)
import numpy as np
import unittest as ut
#from ClassicalNewton import *
from ClassicalNewton import *
from OptimizationProblem import *
def f(x,y):
    return 100*((y-(x**2))**2) + ((1 - x)**2)

def df_dx(x,y):
    return 400*(x**3) - 400*x*y - 2 + 2*x

def df_dy(x,y):
    return 200*y - 200*(x**2)

def df_dx2(x,y):
    return 1200*(x**2) - 400*y + 2

def df_dxy(x,y):
    return -400*x

def df_dy2(x,y):
    return 200

grad = [df_dx,df_dy]

#problem = ClassicalNewton(f,grad)

## Solve the problem with a method here

def manual_hessian(x,y):
    return [[df_dx2(x,y), df_dxy(x,y)],[df_dxy(x,y), df_dy2(x,y)]]
class simpleGradientAndHessianTestCase(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.man_hess = manual_hessian(1,2)
    def testCalculatedGradient(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        gradient = op.get_gradient(f, (1,2))
        np.testing.assert_allclose(gradient, self.man_gradient)
    def testSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        gradient = op.get_gradient(f, (1,2))
        np.testing.assert_allclose(gradient, self.man_gradient)
    def testHessianFunctionWithoutGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        hessian = op.get_hessian(f, (1,2))
        np.testing.assert_allclose(self.man_hess, hessian, 0.0001)
    def testHessianFunctionWithGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        hessian = op.get_hessian(f, (1,2))
        np.testing.assert_allclose(self.man_hess, hessian, 0.0001)
class exactLineSearchTestCase(ut.TestCase):
    def setUp(self):
        self.ros_res = (1,1)
    def testRosenBrockWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve(search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve([2,2], search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithEasyInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        res = op.solve((2, 2), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    
class inexactGoldsteinLineSearchTestCase(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.man_hess = manual_hessian(1,2)
        self.ros_res = (1,1)
    def testRosenBrockWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve(search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve((2,2), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
class inexactWolfePowellLineSearchTestCase(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.man_hess = manual_hessian(1,2)
        self.ros_res = (1,1)
    def testRosenBrockWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve(search='inexact', cond = 'WP')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve((2,2), search='inexact', cond = 'WP')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'WP')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testRosenBrockWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = ClassicalNewton(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'WP')
        np.testing.assert_almost_equal(self.ros_res, res)
if __name__=='__main__':
    ut.main()
