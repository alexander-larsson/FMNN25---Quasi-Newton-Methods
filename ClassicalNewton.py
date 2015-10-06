from OptimizationMethod import *
import scipy.linalg as la
import inspect
from scipy.optimize import minimize_scalar

class ClassicalNewton(OptimizationMethod):

    def solve(self, initial_guess=None):
        return self.classic_newton_method(initial_guess)

    def classic_newton_method(self, initial_guess):
        def gradient_is_zero(gradient):
            return np.allclose(gradient,np.zeros((1,len(gradient))))

	def get_alpha_k(x_k, s_k):
		
            f = self.problem.obj_func
            alpha_k = 1
            minimum = f(*(x_k + s_k)) # alpha = 1
            for alpha in range(1,1000):
                cand = f(*(x_k)) # + alpha*s_k))
                if cand < minimum:
                    minimum = cand
                    alpha_k = alpha
            return alpha_k
	if initial_guess is None:
		x_k = np.zeros(self.problem.obj_func.func_code.co_argcount) 
	else:
		x_k = initial_guess
	gradient = []
        for _ in range(1000):
	    if self.problem.grad is None:
	            gradient = self.get_gradient(self.problem.obj_func,x_k)
	    else:
                    gradient = [g(*x_k) for g in self.problem.grad]  
            hessian = self.get_hessian(self.problem.obj_func,x_k)
            if gradient_is_zero(gradient):
                return x_k
            L = la.cholesky(hessian, lower=True)
            s_k = la.cho_solve((L,True),gradient)
            alpha_k = get_alpha_k(x_k, s_k)
            x_k = x_k - alpha_k*s_k
        raise Exception("Newtons method did not converge")

