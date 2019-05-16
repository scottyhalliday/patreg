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
from distributions import normal, cauchy, gumbel

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

    num_points = 1000

    cdf1 = np.zeros(num_points)
    cdf2 = np.zeros(num_points)
    cdf3 = np.zeros(num_points)
    cdf4 = np.zeros(num_points)

    x1, y1 = normal.normal_distribution(num_points, -20.0, 20.0, 0.0, 1.0)
    x2, y2 = normal.normal_distribution(num_points, -20.0, 20.0, 0.0, 5.0)
    x3, y3 = cauchy.cauchy_distribution(num_points, -20.0, 20.0, 0.0, 0.5)
    x4, y4 = gumbel.gumbel_distribution(num_points, -20.0, 20.0, 0.5, 2.0)


    for i in range(num_points):
        cdf1[i] = cdf(x1[i], x1, y1)
        cdf2[i] = cdf(x2[i], x2, y2)
        cdf3[i] = cdf(x3[i], x3, y3)
        cdf4[i] = cdf(x4[i], x4, y4)

    print(f'CDF for Normal Distribution with mu=0.0, sigma=1.0')
    print(f'CDF at 10.0 is {cdf(10.0,x1,y1)}')
    print(f'CDF at 1.0  is {cdf(1.0,x1,y1)}')
    print(f'CDF at 2.5  is {cdf(2.5,x1,y1)}')
    print('')

    print(f'CDF for Normal Distribution with mu=0.0, sigma=5.0')
    print(f'CDF at 10.0 is {cdf(10.0,x2,y2)}')
    print(f'CDF at 1.0  is {cdf(1.0,x2,y2)}')
    print(f'CDF at 2.5  is {cdf(2.5,x2,y2)}')
    print('')

    print(f'CDF for Cauchy Distribution with xo=0.0, gamma=0.5')
    print(f'CDF at 0.0 is {cdf(0.0,x3,y3)}')
    print(f'CDF at 1.0 is {cdf(1.0,x3,y3)}')
    print(f'CDF at 2.5 is {cdf(2.5,x3,y3)}')
    print('')

    print(f'CDF for Gumbel Distribution with mu=0.5, sigma=2.0')
    print(f'CDF at 0.0 is {cdf(0.0,x4,y4)}')
    print(f'CDF at 1.0 is {cdf(1.0,x4,y4)}')
    print(f'CDF at 2.5 is {cdf(2.5,x4,y4)}')
    print('')

    fig,ax = plt.subplots()
    ax.plot(x1,cdf1, label='Normal - mu=0.0,sigma=1.0')
    ax.plot(x2,cdf2, label='Normal - mu=0.0,sigma=5.0')
    ax.plot(x3,cdf3, label='Cauchy - xo=0.0,gamma=0.5')
    ax.plot(x4,cdf4, label='Gumbel - mu=0.5,sigma=2.0')
    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title("Cumumlative Distribution Function")
    plt.grid()
    plt.show()
