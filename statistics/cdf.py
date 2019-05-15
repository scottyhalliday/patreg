'''
PATREG - A Pattern Recognition Tool Box

cdf.py
Calculates the cumulative distribution function (cdf) of a continuous random
variable is defined by every number x such that for x, F(x) is the area under
the density curve to the left of x.
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
from distributions import normal

def cdf(x: float, xpoints: np.array, ypoints: np.array):
    '''
    x      : Point to calculate cdf to
    xpoints: Array of x-axis coordinates for the distribution
    ypoints: Array of y-axis coordinates for the distribution
    '''
    assert(xpoints.size == ypoints.size)

    P = 0.0
    for i in range(xpoints.size):
        if i == 0:
            P += ypoints[i]
        elif xpoints[i] > x:
            break
        else:
            P += (xpoints[i]-xpoints[i-1])*ypoints[i]

        i+=1

    return P

if __name__=='__main__':
    x1, y1 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 1.0)

    print(f'CDF for Normal Distribution with mu=0.0, sigma=1.0')
    print(f'CDF at 10.0 is {cdf(10.0,x1,y1)}')
    print(f'CDF at 1.0  is {cdf(1.0,x1,y1)}')
    print(f'CDF at 2.5  is {cdf(2.5,x1,y1)}')
    print('')

    x2, y2 = normal.normal_distribution(100, -20.0, 20.0, 0.0, 5.0)
    print(f'CDF for Normal Distribution with mu=0.0, sigma=5.0')
    print(f'CDF at 10.0 is {cdf(10.0,x2,y2)}')
    print(f'CDF at 1.0  is {cdf(1.0,x2,y2)}')
    print(f'CDF at 2.5  is {cdf(2.5,x2,y2)}')
    print('')

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='mu=0,sigma=1')
    ax.plot(x2,y2, label='mu=0,sigma=5')
    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title("Normal Distributions")
    plt.grid()
    plt.show()
