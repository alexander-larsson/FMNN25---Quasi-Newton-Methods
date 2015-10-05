from OptimizationMethod import *
import scipy.linalg as la
from scipy.optimize import minimize_scalar

class ClassicalNewton(OptimizationMethod):

    def solve(self):
        # guess = exact_line_search(self.problem.obj_func,)
        # replace this with line search
        guess = np.array([1,1])
        return self.classic_newton_method(guess)


    def classic_newton_method(self,initial_guess):
        def gradient_is_zero(gradient):
            return np.allclose(gradient,np.zeros((1,len(gradient))))

        xk = initial_guess
        for _ in range(1000):
            gradient = self.get_gradient(self.problem.obj_func,*xk)
            hessian = self.get_hessian(self.problem.obj_func,*xk)
            if gradient_is_zero(gradient):
                return xk

            L = la.cholesky(hessian, lower=True)
            sk = la.cho_solve((L,True),gradient)
            xk = xk - sk
        raise LinAlgError("Newtons method did not converge")

    def exact_line_search(self,function,x_values,s):
        """
        Parameters:
        function = the function
        x_values = the values
        s = newton direction

        Determines alpha(k) by exact linear search(slide : 3.5)
        """
        def f_alpha(alpha):
            return function(x_values+alpha*s)
        return minimize_scalar(f_alpha).x_values

    def test_positive_definiteness(function_degree,hessian):
        """
        Parameters:
        function_degree = the function degree
        hessian = the hessian matrix
        gradient = the gradient

        Raises an LinAlgError(according to the documentation of cho.) :
        If the decomposition fails, for example, if a is not positive-definite.

        """
        factorized = la.cho_factor(hessian)
        solution = la.cho_solve(factorized,function_degree)
        return solution
