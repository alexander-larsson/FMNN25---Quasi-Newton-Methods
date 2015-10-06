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
            print(minimum)
            for alpha in range(1,1000):
                cand = f(*(x_k + alpha*s_k))
                if cand < minimum:
                    minimum = cand
                    alpha_k = alpha
            return alpha_k

        x_k =  np.array([1,0]) #initial guess
        for _ in range(1000):
            gradient = self.get_gradient(self.problem.obj_func,x_k)
            hessian = self.get_hessian(self.problem.obj_func,x_k)
            if gradient_is_zero(gradient):
                return x_k
            L = la.cholesky(hessian, lower=True)
            s_k = la.cho_solve((L,True),gradient)
            alpha_k = get_alpha_k(x_k, s_k)
            x_k = x_k - alpha_k*s_k
        raise Exception("Newtons method did not converge")

