'''
PATREG - A Pattern Recognition Tool Box

Calculates the expected or mean value of a continuous random variable
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
from distributions import normal

def mean(x: np.array, y: np.array):
    assert(x.size == y.size)

    mean = 0.0
    for i in range(x.size):
        #print(f'x={x[i]}, y={y[i]}, x*y={x[i]*y[i]}, mean={mean}')
        if i == 0:
            mean += x[i]*y[i]
        else:
            mean += x[i]*((x[i]-x[i-1])*y[i])
    return mean

if __name__=='__main__':
    x1, y1 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 1.0)
    print(f'Mean for Normal Distribution with mu=0.0, sigma=1.0 is {mean(x1,y1)}')

    x2, y2 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 5.0)
    print(f'Mean for Normal Distribution with mu=0.0, sigma=5.0 is {mean(x2,y2)}')

    x3, y3 = normal.normal_distribution(100, -20.0, 20.0, 2.0, 5.0)
    print(f'Mean for Normal Distribution with mu=2.0, sigma=5.0 is {mean(x3,y3)}')

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='mu=0,sigma=1')
    ax.plot(x2,y2, label='mu=0,sigma=5')
    ax.plot(x3,y3, label='mu=2,sigma=5')
    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title("Normal Distributions")
    plt.grid()
    plt.show()
