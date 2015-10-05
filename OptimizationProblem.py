from OptimizationMethod import *

class OptimizationProblem:
    def __init__(self,objective_funcion,gradient=None):
        """
        Parameters:
        objective_funcion = the function on with to optimize
        gradient = the gradient of the objective function (a list of functions)
        """
        self.obj_func = objective_funcion
        self.om = OptimizationMethod(self.obj_func)
        self.grad = gradient
    def get_gradient_at(self,*point):
        if self.grad is None:
            return self.om.get_gradient(self.obj_func, *point)
        return [g(*point) for g in self.grad]
        
    def get_hessian(self, *point):
        if self.grad is None:
            return self.om.get_hessian(self.obj_func, None, *point)
        return self.om.get_hessian(self.obj_func, self.grad, *point)