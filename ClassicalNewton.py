from OptimizationMethod import *
import scipy.linalg as la
import scipy.optimize as opt

class ClassicalNewton(OptimizationMethod):

    def newton_iteration(self, initial_guess, searchMethod, cond):

        def gradient_is_zero(gradient):
            epsilon = 0.0000001
            return la.norm(gradient) < epsilon

        if initial_guess is None:
            x_k = np.zeros(self.problem.obj_func.__code__.co_argcount)
        else:
            x_k = np.array(initial_guess)
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
            alpha_k = searchMethod(x_k, s_k, cond)
            x_k = x_k - alpha_k*s_k
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        raise Exception("Newtons method did not converge")
