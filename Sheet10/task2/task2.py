import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numdifftools as nd

def banana(x, a = 0, b = 20):
    '''
    banana function
    x is a two dimensional vector
    x = [x0, x1]
    '''
    return (a - x[0])**2 + b * (x[1] - x[0]*x[0])**2

def find_min(foo, x0, gamma = 1., err = 0.00000001, max_iter = 500):
    """
    find minimum by newton method
    x0 = [x,y] start position vector
    """
    x = x0
    converged = False
    steps = 0
    for i in range(max_iter):
        H = nd.Hessian(foo)(x)
        grad = nd.Gradient(foo)(x)
        x = x - gamma * np.dot(np.linalg.inv(H),grad)
        grad = nd.Gradient(foo)(x)
        # if the gradient is small enough in every direction, break
        if (grad[0] < err) & (grad[1] < err):
            converged = True
            steps = i
            break
    
    if converged:
        print("find_min converged after {} steps".format(steps))
        print("the minimum is at {}".format(x))
    return x

def vanilla(foo, x0, gamma = 1., err = 0.00000001, max_iter = 500):
    """
    find minimum by vanilla gradient descent
    x0 = [x,y] start position vector
    """
    x = x0
    converged = False
    steps = max_iter
    for i in range(max_iter):
        grad = nd.Gradient(foo)(x)
        x = x - gamma * grad
        grad = nd.Gradient(foo)(x)
        # if the gradient is small enough in every direction, break
        if (np.abs(grad[0]) < err) & (np.abs(grad[1]) < err):
            converged = True
            steps = i
            break
    if converged:
        print("vanilla gradient descent converged after {} steps".format(steps))
        print("the minimum is at {}".format(x))
    else:
        print("vanilla gradient descent did not converge after {} steps".format(steps))
    return x




print(find_min(banana, [2,2]))
print("--------------------")
print("comparing to vanilla gradient descent:")

print(vanilla(banana, [2,0.0002]))

# For the banana function vanilla gradient descent does not converge
# to the correct minimum, but apparently at that point grad == 0.

# OUTPUT
# find_min converged after 5 steps
# the minimum is at [ 1.75047961e-20 -2.18800833e-16]
# [ 1.75047961e-20 -2.18800833e-16]
# --------------------
# comparing to vanilla gradient descent:
# vanilla gradient descent converged after 2 steps
# the minimum is at [-7.57660805e+32  1.64786768e+07]
# [-7.57660805e+32  1.64786768e+07]
