from OptimizationMethod import *
import scipy.linalg as la
import inspect
import scipy.optimize as opt

class QuasiNewton(OptimizationMethod):
    def newton_iteration(self, initial_guess, searchMethod, cond):
        def gradient_is_zero(gradient):
            epsilon = 0.000001
            return la.norm(gradient) < epsilon

        if initial_guess is None:
            x_k = np.zeros(self.problem.obj_func.__code__.co_argcount)
        else:
            x_k = np.array(initial_guess)
#        hessian = self.get_hessian(self.problem.obj_func, x_k)
#       inv_hessian = np.linalg.inv(hessian)
        inv_hessian = np.eye(len(x_k))
        if self.problem.grad is None:
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        else:
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(100):
            if gradient_is_zero(gradient):
                return x_k
            s_k = np.dot(inv_hessian, gradient)
            alpha_k = searchMethod(x_k, s_k, cond)
            if alpha_k < 0.0000001:
                alpha_k = 0.0000001
            x_k_new = x_k - alpha_k*s_k
            gradient_new = self.get_gradient(self.problem.obj_func,x_k_new)
            delta = (x_k_new - x_k).reshape(len(x_k), 1)
            gamma = (gradient_new - gradient).reshape(len(x_k), 1)
            inv_hessian = self.next_inv_hessian(delta,gamma,inv_hessian)
            gradient = gradient_new
            x_k = x_k_new
        raise Exception("Newtons method did not converge")

class GoodBroyden(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        u = delta - np.dot(inv_hessian,gamma)
        a = 1 / np.dot(u.T,gamma)
        return inv_hessian + a * (np.dot(u,u.T))

class BadBroyden(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        part2 = (gamma - np.dot(inv_hessian, delta))/np.dot(delta.T, delta)
        return inv_hessian + np.dot(part2,delta.T)

class DFP(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        part2 = np.dot(delta, delta.T) / np.dot(delta.T, gamma)
    	part3 = np.dot(inv_hessian, np.dot(gamma, np. dot(gamma.T, inv_hessian))) / np.dot(gamma.T, np.dot(inv_hessian, gamma))
        return inv_hessian + part2 - part3

class BFGS(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        part2 = 1 + np.dot(gamma.T, np.dot(inv_hessian, gamma))/np.dot(delta.T, gamma)
        part3 = delta.dot(delta.T) / np.dot(delta.T,gamma)
        part4 = (np.dot(delta, np.dot(gamma.T , inv_hessian)) + np.dot(inv_hessian, np.dot(gamma,delta.T))) / np.dot(delta.T,gamma)
        return inv_hessian + part2 * part3 - part4
