class OptimizationProblem:

    def __init__(self,objective_funcion,gradient):
        """
        Parameters:
        objective_funcion = the function on with to optimize
        gradient = the gradient of the objective function (a list of functions)
        """
        self.obj_func = objective_funcion
        self.grad = gradient
