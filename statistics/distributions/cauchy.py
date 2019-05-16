'''
PATREG - A Pattern Recognition Tool Box

cauchy.py
Constructs a cauchy distribution over a range of x values.  The Cauchy distribution f(x:xo,gamma) is the distribution
of the x-intercept of a ray issueing from (xo,gamma) with a uniformly distributed angle.

The distribution is calculated based on the following input
xo    - The location parameter specifying the location of the peak of the distribution
gamma - The scale parameter which specifies the half-width at half-maximum
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt


def cauchy_distribution(num_points: int,    # Number of points to evaluate
                        x_min_lim : float,  # X-axis minimum limit
                        x_max_lim : float,  # X-axis maximum limit
                        xo        : float,  # location parameter
                        gamma     : float   # scale parameter
                       ) -> (np.array, np.array):

    cauchy = np.zeros(num_points)

    X=np.linspace(x_min_lim, x_max_lim,num_points)

    for i in range(0,num_points):
        cauchy[i] = (1/(pi*gamma)) * gamma**2/((X[i]-xo)**2 + gamma**2)

    return X,cauchy

if __name__=='__main__':

    x1,y1 = cauchy_distribution(1000, -20.0, 20.0,  0.0, 0.5)
    x2,y2 = cauchy_distribution(1000, -20.0, 20.0,  0.0, 1.0)
    x3,y3 = cauchy_distribution(1000, -20.0, 20.0,  0.0, 2.0)
    x4,y4 = cauchy_distribution(1000, -20.0, 20.0, -2.0, 1.0)

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='xo= 0.0,gamma=0.5')
    ax.plot(x2,y2, label='xo= 0.0,gamma=1.0')
    ax.plot(x3,y3, label='xo= 0.0,gamma=2.0')
    ax.plot(x4,y4, label='xo=-2.0,gamma=1.0')

    ax.set_xlim(left=-5,right=5)
    ax.legend()
    ax.set_title("Cauchy Distributions")
    plt.grid()
    plt.show()
