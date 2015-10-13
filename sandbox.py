from QuasiNewton import *
from OptimizationProblem import *

def Rosenbrock(x,y):
    return 100*((y-(x**2))**2) + ((1 - x)**2)

problem = OptimizationProblem(Rosenbrock)
method = DFP(problem)
print(method.solve((1,0)))
