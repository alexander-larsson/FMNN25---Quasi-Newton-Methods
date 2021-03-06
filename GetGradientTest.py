## Functions to test on
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

def f(x,y):
    return 100*(y-x**2)**2 + (1 - x)**2

def df_dx(x,y):
    return 400*(x**3) - 400*x*y - 2 + 2*x

def df_dy(x,y):
    return 200*y - 200*(x**2)

def df_dx2(x,y):
    return 1200*(x**2) - 400*y + 2

def df_dxy(x,y):
    return -400*x

def df_dy2(x,y):
    return 200

def get_gradient(function,x0,y0):
    """
    Returns the gradient
    """
    res = .05
    x = np.array([x0-res,x0,x0+res])
    y = np.array([y0-res,y0,y0+res])
    X,Y = np.meshgrid(x,y)
    zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    gy,gx = np.gradient(Z,res,res)
    return gx[1][1],gy[1][1]

def get_gradient2(function,*point):
    """
    Gradient for any kind of funtion
    Parameters:
    function = the function
    point = the point where we evaluate the gradient
    """
    res = .05
    n = len(point)
    x = np.empty((n,3))
    for i in range(n):
        xi = point[i]
        x[i] = [xi-res,xi,xi+res]
    X = np.meshgrid(*x)
    zs = np.array([f(*x) for x in zip(*map(np.ravel,X))])
    Z = zs.reshape(X[0].shape)
    gx = np.gradient(Z,res,res)
    # element[1][1] should maybe be something else (only tested for one function)
    result = [element[1][1] for element in reversed(gx)]
    return np.array(result)

def get_gradient3(function,*point):
    """
    Gradient for any kind of funtion
    Parameters:
    function = the function
    point = the point where we evaluate the gradient
    """
    res = 0.000001
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

def get_hessian(function,x0,y0):
    """
    Returns the hessian
    """
    res = .05
    fx = get_gradient2(function,x0,y0)
    fxplush = get_gradient2(function,x0+res,y0)
    fyplush = get_gradient2(function,x0,y0+res)
    fplush = fxplush,fyplush
    hessian = np.empty((2,2))
    for i in range(2):
        for j in range(2):
            hessian[i][j] = (fplush[j][i] - fx[i])/res
    hessian = (hessian + np.transpose(hessian))/2
    return hessian

def get_hessian2(function, *point):
    """
    Calculates the hessian for any kind of function
    Parameters:
    function = the function
    point = the point where we evaluate the hessian
    """
    res = .05
    n = len(point)
    fx = get_gradient2(function,*point)
    fplush = []
    for i in range(n):
        p = list(point)
        p[i] += res
        fplush.append(get_gradient2(function,*p))
    hessian = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            hessian[i][j] = (fplush[j][i] - fx[i])/res
    hessian = (hessian + np.transpose(hessian))/2
    return hessian

def test_hessian(x,y):
    res = .05
    dx2 = (df_dx(x+res,y)-df_dx(x,y))/res
    dxy = (df_dx(x,y+res)-df_dx(x,y))/res
    dyx = (df_dy(x+res,y)-df_dy(x,y))/res
    dy2 = (df_dy(x,y+res)-df_dy(x,y))/res
    print(str(dx2) + " " + str(dxy))
    print(str(dyx) + " " + str(dy2))

def manual_hessian(x,y):
    print(str(df_dx2(x,y)) + " " + str(df_dxy(x,y)))
    print(str(df_dxy(x,y)) + " " + str(df_dy2(x,y)))


#test_hessian(1,2)
#print(get_hessian(f,1,2))
#manual_hessian(1,2)

#print("New hessian")
#print(get_hessian2(f,1,2))

point = (15,15)
print(get_gradient3(f,*point))
print(get_gradient2(f,*point))
print(str(df_dx(*point)) + " " + str(df_dy(*point)))
