# File to test the performance of the Rosenbrock function on the
# classical Newtons method. (Task 5)

from OptimizationProblem import *

def f(x,y):
    return 100*(y-x**2)**2 + (1 - x)**2

def df_dx(x,y):
    return 400*(x**3) - 400*x*y - 2 + 2*x;

def df_dy(x,y):
    return 200*y - 200*(x**2);

grad = [df_dx,df_dy]

problem = OptimizationProblem(f,grad)

## Solve the problem with a method here
