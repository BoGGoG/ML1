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

print(find_min(banana, [2,2]))

# plot 3d function

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-5.0, 5.0, 0.05)
# X, Y = np.meshgrid(x, y)
# zs = np.array([banana(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
