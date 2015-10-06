from OptimizationMethod import *
import scipy.linalg as la
import inspect
import scipy.optimize as opt

class ClassicalNewton(OptimizationMethod):

    def solve(self, initial_guess=None):
        return self.classic_newton_method(initial_guess)

    def classic_newton_method(self, initial_guess):
        def gradient_is_zero(gradient):
            return np.allclose(gradient,np.zeros((1,len(gradient))))

        #x_k =  np.array([1,0]) #initial guess
        if initial_guess is None:
            # This gives me an error now for some reason!!!!!!!!!!! / Alex
            x_k = np.zeros(self.problem.obj_func.func_code.co_argcount)
#           x_k[0] = 1
        else:
            #print("Initial guess not zero!")
            x_k = np.array(initial_guess)
        gradient = []
        if self.problem.grad is None:
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        else:
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(10000):
            hessian = self.get_hessian(self.problem.obj_func,x_k)
            print(gradient)
            if gradient_is_zero(gradient):
                return x_k
            L = la.cholesky(hessian, lower=True)
            s_k = la.cho_solve((L,True),gradient)
            alpha_k = self.exact_line_search(x_k, s_k) # self.problem.grad)
            print("alpha_k = ", alpha_k)
            x_k = x_k - alpha_k*s_k
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        raise Exception("Newtons method did not converge")


    def exact_line_search(self,x_k, s_k):
        """
        Cheat solution, was hungry
        """
        f = self.problem.obj_func
        def alpha_f(alpha):
            return f(*(x_k - alpha*s_k))
        return opt.minimize_scalar(alpha_f).x

    def _create_f_alpha_(self, x_k,s_k):
        def points(alpha):
            return self.problem.obj_func(*(x_k + alpha*s_k))
        return points

    def f_prim(self, f_alpha):
        def val(a):
            res = 1
            return (f_alpha(a) + f_alpha(a + res)) / res
        return val

    def inexact_line_search(self, x_k, s_k, grad):
        rho = 0.1
        sigma = 0.7
        tau = 0.1
        chi = 9
        a_0 = 1
        f_alpha = self._create_f_alpha_(x_k,s_k)
        f_grad = self.f_prim(f_alpha)
        def extrapolation(a_0,a_l):
            return ((a_0 - a_l)*f_grad(a_0)) / (f_grad(a_l) - f_grad(a_0))
        def interpolation(a_0,a_l):
            gradient_l = f_grad(a_l)
            f_l = f_alpha(a_l)
            f_0 = f_alpha(a_0)
            return (((a_0 - a_l)**2) * gradient_l) / (2*( f_l - f_0 + (a_0-a_l)*gradient_l))

        def LC(a_0, a_l):
            print("LC: f_alpha(a_0) = ", f_alpha(a_0))
            print("LC: f_alpha(a_l) = ", f_alpha(a_l))
            print("LC: f_grad(a_l) = ", f_grad(a_l))
            print("LC: stuff = ", (1-rho)*(a_0 - a_l)*f_grad(a_l))
            print("LC: other stuff = ", f_alpha(a_l) + ((1-rho)*(a_0 - a_l)*f_grad(a_l)))
            return f_alpha(a_0) >= f_alpha(a_l) + ((1-rho)*(a_0 - a_l)*f_grad(a_l))

        def RC(a_0, a_l):
            return f_alpha(a_0) <= f_alpha(a_l) + rho*(a_0 - a_l)*f_grad(a_l)
        a_l = 0 #This works if it's at one but not if it's at zero. It should be a zero. =(
        a_u = 10**99
        lc = LC(a_0, a_l)
        rc = RC(a_0, a_l)
        while not (lc and rc):
            if lc:
                print("LC")
                d_alpha_0 = extrapolation(a_0, a_l)
                d_alpha_0 = max(d_alpha_0,tau*(a_0 - a_l))
                d_alpha_0 = min(d_alpha_0,chi*(a_0 - a_l))
                a_l = a_0
                a_0 = a_0 + d_alpha_0
            else:
                print("RC")
                a_u = min(a_0, a_u)
                str_alpha_0 = interpolation(a_0, a_l)
                str_alpha_0 = max(str_alpha_0, (a_l + tau*(a_u - a_l)))
                str_alpha_0 = min(str_alpha_0, (a_u - tau*(a_u - a_l)))
                print("str_alpha_0 = ", str_alpha_0)
                a_0 = str_alpha_0
            lc = LC(a_0, a_l)
            rc = RC(a_0, a_l)
        return a_0
