'''
PATREG - A Pattern Recognition Tool Box

gumbel.py
Constructs a Gumbel distribution given the mean (mu) and standard deviation
(sigma).  This is used to model the distribution of the maximum (or the
minimum) of a number of samples of various distributions.  This distribution
might be used to represent the distribution of the maximum level of a river in
a particular year if there was a list of maximum values for the past ten years.
It is useful in predicting the chance of extreme earthquake, flood or other
natural disaster will occur. (--Wikipedia)
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

def gumbel_distribution(num_points: int,    # Number of points to evaluate
                        x_min_lim : float,  # X-axis minimum limit
                        x_max_lim : float,  # X-axis maximum limit
                        mu        : float,  # Mode
                        sigma     : float   # Standard Deviation
                       ) -> (np.array, np.array):

    ndist = np.zeros(num_points)

    X=np.linspace(x_min_lim, x_max_lim,num_points)

    beta = 0.78*sigma

    for i in range(0,num_points):
        z = (X[i]-mu)/beta
        ndist[i] = 1/beta * exp(-1.0*(z+exp(-z)))

    return X,ndist


if __name__=='__main__':
    x1, y1 = gumbel_distribution(1000, -20.0, 20.0, 0.5, 2.0)
    x2, y2 = gumbel_distribution(1000, -20.0, 20.0, 1.0, 2.0)
    x3, y3 = gumbel_distribution(1000, -20.0, 20.0, 1.5, 3.0)
    x4, y4 = gumbel_distribution(1000, -20.0, 20.0, 3.0, 4.0)

    fig,ax = plt.subplots()
    ax.plot(x1,y1 , label='mu=0.5,sigma=2.0')
    ax.plot(x2,y2 , label='mu=1.0,sigma=2.0')
    ax.plot(x3,y3 , label='mu=1.5,sigma=3.0')
    ax.plot(x4,y4 , label='mu=3.0,sigma=4.0')

    ax.set_xlim(left=-5,right=20)
    ax.legend()
    ax.set_title("Gumbel Distributions")
    plt.grid()
    plt.show()
