import numpy as np
import scipy.optimize as opt

class OptimizationMethod:
    """
    A general Optimization Method class. Other methods should
    inherit from this class and the common operations should be
    put here to avoid duplicated code.
    """

    def __init__(self,optimization_problem):
        self.problem = optimization_problem
        self.res = 0.000001

    def solve(self, initial_guess=None, search="inexact", cond="GS"):
        if (search.lower() == "inexact"):
            searchMethod = self.inexact_line_search
        elif search.lower() == "exact":
            searchMethod = self.exact_line_search
        else:
            raise TypeError("search must be one of 'exact' or 'inexact'")
        return self.newton_iteration(initial_guess, searchMethod, cond)

    def get_gradient(self,function, point):
        """
        Gradient for any kind of funtion
        Parameters:
        function = the function
        point = the point where we evaluate the gradient
        """
        #res = self.res
        res = 0.000003
        n = len(point)
        gradient = np.empty(n)
        for i in range(n):
            x = list(point) # Make a copy
            x[i] += res/2
            fplush = function(*x)
            x[i] -= res
            fminush = function(*x)
            gradient[i] = (fplush - fminush)/res
        return gradient

    def get_hessian(self, function, point, grad=None):
        """
        Hessian for any function
        Parameters:
        function = the function
        point = the point where we evaluate the hessian
        """
        res = self.res
        n = len(point)
        if grad is None:
            fx = self.get_gradient(function,point)
        else:
            fx = [g(point) for g in grad]
        fplush = []
        for i in range(n):
            p = list(point)
            p[i] += res
            if grad is None:
                fplush.append(self.get_gradient(function,p))
            else:
                fplush.append([g(p) for g in grad])
        hessian = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                hessian[i][j] = (fplush[j][i] - fx[i])/res
        hessian = (hessian + np.transpose(hessian))/2
        return hessian

    def exact_line_search(self,x_k, s_k, cond=None):
        """While exact line search doesn't care about a condition, we include
        it in order to ease up the manipulation of the line search method"""
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
            res = self.res
            return (f_alpha(a+res) - f_alpha(a)) /res
        return val

    def inexact_line_search(self, x_k, s_k, cond, r=0.1, s = 0.7, t=0.1, c = 9):
        #Helper functions
        def LCG(a_0, a_l):
            return f_alpha(a_0) >= (f_alpha(a_l) + (1-rho)*(a_0 - a_l)*f_grad(a_l))
        def RCG(a_0, a_l):
            return f_alpha(a_0) <= (f_alpha(a_l) + rho*(a_0 - a_l)*f_grad(a_l))
        def LCWP(a_0, a_l):
            return f_grad(a_0) >= sigma*f_grad(a_l)
        def RCWP(a_0, a_l):
            return f_alpha(a_0) <= f_alpha(a_l) + rho*(a_0 - a_l)*f_grad(a_l)
        def extrapolation(a_0,a_l):
            A = f_grad(a_l) - f_grad(a_0)
            #Hack
            #if A < 0.00000001:
            #    A = 0.00000001
            return (a_0 - a_l)*(f_grad(a_0) / A)
        def interpolation(a_0,a_l):
            gradient_l = f_grad(a_l)
            f_l = f_alpha(a_l)
            f_0 = f_alpha(a_0)
            return (a_0 - a_l)**2*gradient_l / (2*(f_l - f_0 + (a_0 - a_l)*gradient_l))
        #Define initial conditions and contants

        if (cond.upper() == 'GS'):
            LC, RC = LCG, RCG
        elif cond.upper() == 'WP':
            LC, RC = LCWP, RCWP
        else:
            raise TypeError("Cond must be one of 'GS' or 'WP'")
        rho = r
        sigma = s
        tau = t
        chi = c
        a_0 = 1
        f_alpha = self._create_f_alpha_(x_k,s_k)
        f_grad = self.f_prim(f_alpha)


        a_l = 0
        a_u = 10**99
        lc = LC(a_0, a_l)
        rc = RC(a_0, a_l)
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
            lc = LC(a_0, a_l)
            rc = RC(a_0, a_l)
        return a_0
