## Functions to test on
import numpy as np

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

## get_gradient(f,x,y) gives almost the same result as ( df_dy(x,y), df_dx(x,y) )

def get_gradient(function,x0,y0):
    """
    Only works for 2-paramter functions.
    Might be able to generalize it.
    """
    res = .05
    x = np.array([x0-res,x0,x0+res])
    y = np.array([y0-res,y0,y0+res])
    X,Y = np.meshgrid(x,y)
    zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    gy,gx = np.gradient(Z,res,res)
    return gx[1][1],gy[1][1]

def get_hessian(function,x0,y0):
    """
    Assumes 2-paramter function
    OMG this actually works!!!
    """
    res = .05
    fx = get_gradient(function,x0,y0)
    fxplush = get_gradient(function,x0+res,y0)
    fyplush = get_gradient(function,x0,y0+res)
    fplush = fxplush,fyplush
    hessian = np.empty((2,2))
    for i in range(2):
        for j in range(2):
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


test_hessian(1,2)
print(get_hessian(f,1,2))
manual_hessian(1,2)
