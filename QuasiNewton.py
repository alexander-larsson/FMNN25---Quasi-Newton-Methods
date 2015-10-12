from OptimizationMethod import *
import scipy.linalg as la
import inspect
import scipy.optimize as opt

class QuasiNewton(OptimizationMethod):

    def newton_iteration(self, initial_guess, searchMethod, cond):
        def gradient_is_zero(gradient):
            epsilon = 0.00000001
            return np.all(list(map(lambda x: np.abs(x) < epsilon,gradient)))

        if initial_guess is None:
            x_k = np.zeros(self.problem.obj_func.__code__.co_argcount)
        else:
            x_k = np.array(initial_guess)
        hessian = self.get_hessian(self.problem.obj_func,x_k)
        inv_hessian = np.linalg.inv(hessian)
        if self.problem.grad is None:
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        else:
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(10000):
            if gradient_is_zero(gradient):
                return x_k
            #L = la.cholesky(hessian, lower=True)
            #s_k = la.cho_solve((L,True),gradient)
            s_k = np.dot(inv_hessian, gradient)
            alpha_k = searchMethod(x_k, s_k, cond)
            x_k_old = x_k;
            x_k = x_k - alpha_k*s_k
            delta = (x_k - x_k_old)
            gamma = (self.get_gradient(self.problem.obj_func,x_k_old)) - (self.get_gradient(self.problem.obj_func,x_k))
            hessian = self.goodBroyden(delta,gamma,hessian)
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        raise Exception("Newtons method did not converge")

    def goodBroyden(self,delta,gamma,hessian):
        u = delta - np.dot(hessian,gamma)
        #a = 1/(u.dot(gamma))
        a = 1 / np.dot(np.transpose(u),gamma)
        hessian = hessian + a * np.dot(u,np.transpose(u))
        return hessian
