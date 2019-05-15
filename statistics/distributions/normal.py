'''
PATREG - A Pattern Recognition Tool Box

normal.py
Constructs a normal distribution given the mean (mu) and standard deviation
(sigma).
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

def normal_distribution(num_points: int,    # Number of points to evaluate
                        x_min_lim : float,  # X-axis minimum limit
                        x_max_lim : float,  # X-axis maximum limit
                        mu        : float,  # Mean
                        sigma     : float   # Standard Deviation
                       ) -> (np.array, np.array):

    ndist = np.zeros(num_points)

    X=np.linspace(x_min_lim, x_max_lim,num_points)

    for i in range(0,num_points):
        ndist[i] = 1/sqrt(2*pi*(sigma**2))*exp(-(X[i]-mu)**2/(2*sigma**2))

    return X,ndist


if __name__=='__main__':
    x1, std_norm = normal_distribution(100, -20.0, 20.0, 0.0, 1.0)
    x2, norm1    = normal_distribution(100, -20.0, 20.0, 1.0, 1.0)
    x3, norm2    = normal_distribution(100, -20.0, 20.0, 1.0, 2.0)
    x4, norm3    = normal_distribution(100, -20.0, 20.0, 2.0, 1.0)
    x5, norm4    = normal_distribution(100, -20.0, 20.0, 2.0, 2.0)

    fig,ax = plt.subplots()
    ax.plot(x1,std_norm, label='mu=0,sigma=1')
    ax.plot(x2,norm1   , label='mu=1,sigma=1')
    ax.plot(x3,norm2   , label='mu=1,sigma=2')
    ax.plot(x4,norm3   , label='mu=2,sigma=1')
    ax.plot(x5,norm4   , label='mu=2,sigma=2')

    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title("Normal Distributions")
    plt.grid()
    plt.show()
