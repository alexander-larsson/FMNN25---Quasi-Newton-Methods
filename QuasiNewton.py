from OptimizationMethod import *
import scipy.linalg as la
import inspect
import scipy.optimize as opt

class QuasiNewton(OptimizationMethod):

    def newton_iteration(self, initial_guess, searchMethod, cond):

        def gradient_is_zero(gradient):
            epsilon = 0.0000001
            return la.norm(gradient) < epsilon

        if initial_guess is None:
            x_k = np.zeros(self.problem.obj_func.__code__.co_argcount)
        else:
            x_k = np.array(initial_guess)
        hessian = self.get_hessian(self.problem.obj_func,x_k)
        inv_hessian = np.linalg.inv(hessian)
        if self.problem.grad is None:
            gradient = self.get_gradient(self.problem.obj_func,x_k)
        else:
            ## I dont account for this case in the code below
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(10000):
            if gradient_is_zero(gradient):
                return x_k
            s_k = np.dot(inv_hessian, gradient)
            alpha_k = searchMethod(x_k, s_k, cond)
            ## Hack
            if alpha_k < 0.0000001:
                alpha_k = 0.0000001
            x_k_new = x_k - alpha_k*s_k
            ## Hack
            #if np.all(x_k_new == x_k):
            #    x_k_new += 0.0000001
            print("xk",x_k,"new xk",x_k_new)
            gradient_new = self.get_gradient(self.problem.obj_func,x_k_new)
            delta = (np.array(x_k_new) - np.array(x_k)).reshape(len(x_k), 1)
            print("grad",gradient,"new gradient",gradient_new)
            gamma = (gradient_new - gradient).reshape(len(x_k), 1)
            inv_hessian = self.next_inv_hessian(delta,gamma,inv_hessian)
            gradient = gradient_new
            x_k = x_k_new
        raise Exception("Newtons method did not converge")

class GoodBroyden((QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        # Good broyden
        # Will make this work then move to other class
        u = delta - np.dot(inv_hessian,gamma)
        #a = 1/(u.dot(gamma))
        print("u",u.T,"gamma",gamma)
        a = 1 / np.dot(u.T,gamma)
        return inv_hessian + a * np.dot(u,u.T)

class BadBroyden(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
        return inv_hessian + np.dot(((delta - np.dot(inv_hessian,gamma)) / np.dot(gamma.T,gamma)),gamma.T)

class DFP(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
#        part2 = (np.dot(delta,delta.T)) / (np.dot(delta.T,gamma))
#part3 = (inv_hessian*np.dot(delta,delta.T)*inv_hessian) / (np.dot(gamma.T,inv_hessian*gamma))
        part2 = delta.dot(delta.T) / delta.T.dot(gamma)
	part3 = inv_hessian.dot(delta).dot(delta.T).dot(inv_hessian) / gamma.T.dot(inv_hessian).dot(gamma)
        return inv_hessian + part2 + part3

class BFGS(QuasiNewton):
    def next_inv_hessian(self,delta,gamma,inv_hessian):
# part2 = 1. + np.dot(np.dot(gamma.T,inv_hessian),gamma)/(np.dot(delta.T,gamma))
	part2 = 1. + gamma.dot(gamma.T).dot(inv_hessian)/ delta.T.dot(gamma)
        part3 = delta,delta.T) / np.dot(delta.T,gamma)
        part4 = (np.dot(np.dot(delta,gamma.T),inv_hessian) + np.dot(np.dot(inv_hessian,gamma),delta.T)) / np.dot(delta.T,gamma)
        return inv_hessian + part2 * part3 - part4
