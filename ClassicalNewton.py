from OptimizationMethod import *
import scipy.linalg as la
import inspect
import scipy.optimize as opt

class ClassicalNewton(OptimizationMethod):

    def solve(self, initial_guess=None):
        return self.classic_newton_method(initial_guess)

    def classic_newton_method(self, initial_guess):
        def gradient_is_zero(gradient):
            epsilon = 0.00000001
            return np.all(list(map(lambda x: np.abs(x) < epsilon,gradient)))

        if initial_guess is None:
            x_k = np.zeros(self.problem.obj_func.__code__.co_argcount)
        else:
            x_k = np.array(initial_guess)
        gradient = []
        if self.problem.grad is None:
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        else:
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(10000):
            hessian = self.get_hessian(self.problem.obj_func,x_k)
            if gradient_is_zero(gradient):
                return x_k
            L = la.cholesky(hessian, lower=True)
            s_k = la.cho_solve((L,True),gradient)
            #alpha_k = self.exact_line_search(x_k, s_k)
            alpha_k = self.inexact_line_search(x_k, s_k)
            x_k = x_k - alpha_k*s_k
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        raise Exception("Newtons method did not converge")


    def exact_line_search(self,x_k, s_k):
        f = self.problem.obj_func
        def alpha_f(alpha):
            return f(*(x_k - alpha*s_k))
        return opt.minimize_scalar(alpha_f).x


    def _create_f_alpha_(self, x_k,s_k):
        def points(alpha):
            return self.problem.obj_func(*(x_k - alpha*s_k))
        return points

    def f_prim(self, f_alpha):
        def val(a):
            res = 0.001
            return (f_alpha(a+res) - f_alpha(a)) /res

        return val

    def inexact_line_search(self, x_k, s_k):
        rho = 0.1
        sigma = 0.7
        tau = 0.1
        chi = 9
        a_0 = 1
        f_alpha = self._create_f_alpha_(x_k,s_k)
        f_grad = self.f_prim(f_alpha)
        def extrapolation(a_0,a_l):
            return (a_0 - a_l)*(f_grad(a_0) / (f_grad(a_l) - f_grad(a_0)))
        def interpolation(a_0,a_l):
            gradient_l = f_grad(a_l)
            f_l = f_alpha(a_l)
            f_0 = f_alpha(a_0)
            return (a_0 - a_l)**2*gradient_l / (2*(f_l - f_0 + (a_0 - a_l)*gradient_l))
        def LCG(a_0, a_l):
            return f_alpha(a_0) >= (f_alpha(a_l) + (1-rho)*(a_0 - a_l)*f_grad(a_l))
        def RCG(a_0, a_l):
            return f_alpha(a_0) <= (f_alpha(a_l) + rho*(a_0 - a_l)*f_grad(a_l))
        def LCWP(a_0, a_l):
            return f_grad(a_0) >= sigma*f_grad(a_l)
        
        def RCWP(a_0, a_l):
            return f_alpha(a_0) <= f_alpha(a_l) + rho*(a_0 - a_l)*f_grad(a_l)

        a_l = 0
        a_u = 10**99
        lc = LCWP(a_0, a_l)
        rc = RCWP(a_0, a_l)
        while not (lc and rc):
            if not lc:
                d_alpha_0 = extrapolation(a_0, a_l)
                d_alpha_0 = np.max([d_alpha_0,tau*(a_0 - a_l)])
                d_alpha_0 = np.min([d_alpha_0,chi*(a_0 - a_l)])
                a_l = a_0
                a_0 = a_0 + d_alpha_0
            else:
                a_u = np.min([a_0,a_u])
                str_alpha_0 = interpolation(a_0, a_l)
                str_alpha_0 = np.max([str_alpha_0,a_l + tau*(a_u-a_l)])
                str_alpha_0 = np.min([str_alpha_0,a_u - tau*(a_u - a_l)])
                a_0 = str_alpha_0
            lc = LCWP(a_0, a_l)
            rc = RCWP(a_0, a_l)
        return a_0

