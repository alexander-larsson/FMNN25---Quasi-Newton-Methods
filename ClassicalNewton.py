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

        x_k =  np.array([1,0]) #initial guess

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
            alpha_k = self.exact_line_search(x_k, s_k)
            x_k = x_k - alpha_k*s_k
        raise Exception("Newtons method did not converge")


    def exact_line_search(self,x_k, s_k):
        f = self.problem.obj_func
        alpha_k = 1
        minimum = f(*(x_k + s_k)) # alpha = 1
        print(minimum)
        for alpha in range(1,1000):
            cand = f(*(x_k + alpha*s_k))
            if cand < minimum:
                minimum = cand
                alpha_k = alpha
        return alpha_k


    def _create_f_alpha_(self,x_k,s_k):
        def points(alpha):
            return x_k + alpha*s_k
        return points



    def inexact_line_search(self, x_k, s_k):
        rho = 0.1
        sigma = 0.7
        tau = 0.1
        chi = 9
        get_alpha_points = self._create_f_alpha_(x_k,s_k)
        def extrapolation( alpha_l,alpha_0):
            gradient_0 = self.get_gradient(self.problem.obj_func, get_alpha_points(alpha_0))
            gradient_l = self.get_gradient(self.problem.obj_func, get_alpha_points(alpha_l))
            return (alpha_0 - alpha_l)* gradient_0 / ( gradient_l - gradient_0 )

        def interpolation(self, alpha_l,alpha_0):
            gradient_0 = self.get_gradient(self.problem.obj_func, get_alpha_points(alpha_0))
            gradient_l = self.get_gradient(self.problem.obj_func, get_alpha_points(alpha_l))
            f_l = self.problem.obj_func(get_alpha_points(alpha_l))
            f_0 = self.problem.obj_func(get_alpha_points(alpha_0))
            return (alpha_0 - alpha_l)**2 * gradient_0 / 2*( f_l - f_0 + (alpha_0-alpha_l)*gradient_l )

        def block1(alpha_l,  alpha_0 , alpha_u):
            d_alpha_0 = extrapolation(alpha_l, alpha_0)
            d_alpha_0 = max(d_alpha_0,tau*(alpha_0 - alpha_l))
            d_alpha_0 = min(d_alpha_0,chi*(alpha_0 - alpha_l))
            alpha_l = alpha_0
            alpha_0 = alpha_0 + d_alpha_0
            return alpha_l, alpha_0, alpha_u
        def block2(alpha_l,  alpha_0 ,alpha_u):
            alpha_u = min(alpha_0, alpha_u)
            str_alpha_0 = interpolation(alpha_l, alpha_0)
            str_alpha_0 = max( str_alpha_0, alpha_l + tau*(alpha_u - alpha_l) )
            str_alpha_0 = min( str_alpha_0, alpha_u - tau*(alpha_u - alpha_l) )
            alpha_0 = str_alpha_0
            return alpha_l, alpha_0, alpha_u
        def LC():
            return false
        def RC():
            return false


        LC = ...
        RC = ...

        while not (LC and RC):
            if LC:
                block1
            else:
                block2

            compute stuff
        retrun alpha_0 and f(alpha_0)
