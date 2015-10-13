from QuasiNewton import *
from OptimizationProblem import *

def Rosenbrock(x,y):
    return 100*((y-(x**2))**2) + ((1 - x)**2)

problem = OptimizationProblem(Rosenbrock)
"""
method = BFGS(problem)
print(method.solve((-1.2,1)))
	"""
method = DFP(problem)
print(method.solve((-1.2,1)))

