## Functions to test on
import numpy as np

def f(x,y):
    return 100*(y-x**2)**2 + (1 - x)**2

def df_dx(x,y):
    return 400*(x**3) - 400*x*y - 2 + 2*x;

def df_dy(x,y):
    return 200*y - 200*(x**2);

## get_gradient(f,x,y) gives almost the same result as ( df_dy(x,y), df_dx(x,y) )

def get_gradient(function,x0,y0):
    """
    Only works for 2-dimendional functions.
    Might be able to generalize it.
    """
    res = .05
    x = np.array([x0-res,x0,x0+res])
    y = np.array([y0-res,y0,y0+res])
    X,Y = np.meshgrid(x,y)
    zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    gy,gx = np.gradient(Z,res,res)
    return gy[1][1],gx[1][1]

print(df_dy(1,2))
print(df_dx(1,2))

print(get_gradient(f,1,2))
