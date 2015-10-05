from OptimizationMethod import *
import scipy.linalg as la
from scipy.optimize import minimize_scalar

class ClassicalNewton(OptimizationMethod):

    def solve(self):
        return self.classic_newton_method()

    def classic_newton_method(self):
        def gradient_is_zero(gradient):
            return np.allclose(gradient,np.zeros((1,len(gradient))))

        def get_alpha_k(x_k, s_k):
            f = self.problem.obj_func
            alpha_k = 1
            minimum = f(*x_k) # alpha = 0
            for alpha in range(1,1000):
                cand = f(*(x_k + alpha*s_k))
                if cand < minimum:
                    minimum = cand
                    alpha_k = alpha
            return alpha_k

        x_k =  np.array([0,0]) #initial guess
        for _ in range(1000):
            gradient = self.get_gradient(self.problem.obj_func,*x_k)
            hessian = self.get_hessian(self.problem.obj_func,*x_k)
            if gradient_is_zero(gradient):
                return x_k
            L = la.cholesky(hessian, lower=True)
            s_k = la.cho_solve((L,True),gradient)
            alpha_k = get_alpha_k(x_k, s_k)
            x_k = x_k - alpha_k*s_k
        raise Exception("Newtons method did not converge")

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
