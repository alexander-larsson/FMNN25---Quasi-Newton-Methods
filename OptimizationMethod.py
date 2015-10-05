import numpy as np

class OptimizationMethod:
    """
    A general Optimization Method class. Other methods should
    inherit from this class and the common operations should be
    put here to avoid duplicated code.
    """

    def __init__(self,optimization_problem):
        self.problem = optimization_problem
        
    
    def get_gradient(self, function, *point):
        """
        Gradient for any kind of funtion
        Parameters:
        function = the function
        point = the point where we evaluate the gradient
        """
        
        res = .000005
        n = len(point)
        x = np.empty((n,3))
        for i in range(n):
            xi = point[i]
            x[i] = [xi-res,xi,xi+res]
        X = np.meshgrid(*x)
        zs = np.array([function(*x) for x in zip(*map(np.ravel,X))])
        Z = zs.reshape(X[0].shape)
        gx = np.gradient(Z,res,res)
        result = [element[1][1] for element in reversed(gx)]
        return np.array(result)
        
    def get_hessian(self, function, grad, *point):
        res = .0000005
        n = len(point)
        print("Optimization_method, point:", point)
        if grad is None:
            fx = self.get_gradient(function,*point)
        else:
            fx = [g(*point) for g in grad]
        fplush = []
        for i in range(n):
            p = list(point)
            #old_p = p[i]
            p[i] += res
            if grad is None:
                fplush.append(self.get_gradient(function,*p))
            else:
                fplush.append([g(*p) for g in grad])
            #p[i] = old_p
        hessian = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                hessian[i][j] = (fplush[j][i] - fx[i])/res
        hessian = (hessian + np.transpose(hessian))/2
        return hessian