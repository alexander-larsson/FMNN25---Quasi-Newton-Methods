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
            ## I dont account for this case in the code below
            gradient = [g(*x_k) for g in self.problem.grad]
        for _ in range(10000):
            if gradient_is_zero(gradient):
                return x_k
            #L = la.cholesky(hessian, lower=True)
            #s_k = la.cho_solve((L,True),gradient)
            s_k = np.dot(inv_hessian, gradient)
            alpha_k = searchMethod(x_k, s_k, cond)
            print("x_k",x_k)
            print("alpha_k",alpha_k)
            x_k_new = x_k - alpha_k*s_k
            print("x_k_new",x_k_new)
            gradient_new = self.get_gradient(self.problem.obj_func,x_k_new)
            #delta = x_k_new - x_k
            delta = (np.array(x_k_new) - np.array(x_k)).reshape(len(x_k), 1)
            print("grad",gradient)
            print("grad_new",gradient_new)
            #gamma = gradient - gradient_new
            gamma = (gradient_new - gradient).reshape(len(x_k), 1)
            inv_hessian = self.next_inv_hessian(delta,gamma,inv_hessian)
            gradient = gradient_new
            x_k = x_k_new
        raise Exception("Newtons method did not converge")

    def next_inv_hessian(self,delta,gamma,inv_hessian):
        # Good broyden
        # Will make this work then move to other class
        print("delta",delta)
        print("gamma",gamma)
        u = delta - np.dot(inv_hessian,gamma)
        print("u",u)
        #a = 1/(u.dot(gamma))
        a = 1 / np.dot(np.transpose(u),gamma)
        return inv_hessian + a * np.dot(u,np.transpose(u))
