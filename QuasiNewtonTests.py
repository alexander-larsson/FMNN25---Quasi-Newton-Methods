# File to test the performance of the f function on the
# classical Newtons method. (Task 5)
import numpy as np
import unittest as ut
#from ClassicalNewton import *
from QuasiNewton import *
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
class goodBroydenTest(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.ros_res = (1,1)
    def testGoodBryodenWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve(search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve([2,2], search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithEasyInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = GoodBroyden(problem)
        res = op.solve((2, 2), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = GoodBroyden(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithStandardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve(search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithEasyInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve((2,2), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithHardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = GoodBroyden(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testGoodBroydenWithHardInitialGuessAndSuppliedGradientInexact(self):
        problem = OptimizationProblem(f, grad)
        op = GoodBroyden(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
class badBroydenTest(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.ros_res = (1,1)
    def testBadBroydennWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve(search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve([2,2], search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithEasyInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = BadBroyden(problem)
        res = op.solve((2, 2), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = BadBroyden(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithStandardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve(search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithEasyInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve((2,2), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithHardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BadBroyden(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBadBroydenWithHardInitialGuessAndSuppliedGradientInexact(self):
        problem = OptimizationProblem(f, grad)
        op = BadBroyden(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
class DFPTest(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.ros_res = (1,1)
    def testDFPWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve(search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve([2,2], search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithEasyInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = DFP(problem)
        res = op.solve((2, 2), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = DFP(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithStandardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve(search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithEasyInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve((2,2), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithHardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = DFP(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testDFPWithHardInitialGuessAndSuppliedGradientInexact(self):
        problem = OptimizationProblem(f, grad)
        op = DFP(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
class BFGSTest(ut.TestCase):
    def setUp(self):
        self.man_gradient = [df_dx(1,2), df_dy(1,2)]
        self.ros_res = (1,1)
    def testBFGSWithStandardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve(search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithEasyInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve([2,2], search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithEasyInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = BFGS(problem)
        res = op.solve((2, 2), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithHardInitialGuess(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithHardInitialGuessAndSuppliedGradient(self):
        problem = OptimizationProblem(f, grad)
        op = BFGS(problem)
        res = op.solve((-1.2, 1), search='exact')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithStandardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve(search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithEasyInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve((2,2), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithHardInitialGuessInexact(self):
        problem = OptimizationProblem(f)
        op = BFGS(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)
    def testBFGSWithHardInitialGuessAndSuppliedGradientInexact(self):
        problem = OptimizationProblem(f, grad)
        op = BFGS(problem)
        res = op.solve((-1.2, 1), search='inexact', cond = 'GS')
        np.testing.assert_almost_equal(self.ros_res, res)

if __name__=='__main__':
    ut.main()
