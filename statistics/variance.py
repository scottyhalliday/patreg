'''
PATREG - A Pattern Recognition Tool Box

Calculates the variance and standard deviation of a continuous random variable
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
import mean
from distributions import normal

def variance(mu: float, x: np.array, y: np.array):
    assert(x.size == y.size)

    sigma2 = 0
    for i in range(x.size):
        if i == 0:
            sigma2 += (x[i]*y[i])*(x[i]-mu)**2
        else:
            sigma2 += ((x[i]-x[i-1])*y[i])*(x[i]-mu)**2
    return sigma2, sqrt(sigma2)

if __name__=='__main__':
    x1, y1 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 1.0)
    print(f'Variance and Standard Deviation for Normal Distribution with mu=0.0, sigma=1.0 is {variance(mean.mean(x1,y1),x1,y1)}')

    x2, y2 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 5.0)
    print(f'Variance and Standard Deviation for Normal Distribution with mu=0.0, sigma=5.0 is {variance(mean.mean(x2,y2),x2,y2)}')

    x3, y3 = normal.normal_distribution(100, -20.0, 20.0, 2.0, 5.0)
    print(f'Variance and Standard Deviation for Normal Distribution with mu=2.0, sigma=5.0 is {variance(mean.mean(x3,y3),x3,y3)}')

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='mu=0,sigma=1')
    ax.plot(x2,y2, label='mu=0,sigma=5')
    ax.plot(x3,y3, label='mu=2,sigma=5')
    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title("Normal Distributions")
    plt.grid()
    plt.show()
