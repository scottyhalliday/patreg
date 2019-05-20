'''
PATREG - A Pattern Recognition Tool Box

pdf.py
Calculates the probability density function (pdf) of a continuous random variable
such that for any two numbers 'a' and 'b' with a<=b.  That is, the probability
that the continuous random variable takes on a value in the interval [a,b] is
the area under the graph of the density function.

CAUTION: Make sure you supply enough data points to represent the curve.  There
are accuracy issues integrating over curves defined with few data points.  To
determine the number of data points needed, run this method with increasing
number of data points until the solution stops changing.
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
from patreg.statistics.distributions import normal

def pdf(a: float, b: float, x: np.array, y: np.array) -> float:
    '''
    Calculate the probability that the value exists between a and b.
        a - X-axis minimum limit
        b - X-axis maximum limit
        x - X-coordinates for distribution
        y - Y-coordinates for distribution
    Note: The sizes of x and y must be the same or an error is thrown
    '''
    # Make sure that the distribution arrays for x and y are same size
    assert(x.size == y.size)

    # Calculate the probability
    P = 0

    # If looking for exact probability then discretize the curve to approximate
    # the value.  This calculates the area under the curve for the y value with
    # an x-value of one.
    if a == b:
        a -= 0.5
        b += 0.5

    # If an interval is requested then perform the integration
    for i in range(x.size):

        # Calculate probability only when we are within the limits specified.
        if x[i] < a:
            continue
        if x[i] > b:
            break

        if i > 0:
            P += (x[i]-x[i-1])*y[i]
        else:
            P += y[i]

    return P

if __name__=='__main__':
    x1, y1 = normal.normal_distribution(5000, -20.0, 20.0, 0.0, 1.0)
    print(f'PDF for Normal Distribution with mu=0.0, sigma=1.0')
    print(f'PDF for interval -1   to 1   is {pdf(-1,1,x1,y1)}')
    print(f'PDF for interval -2   to 2   is {pdf(-2,2,x1,y1)}')
    print(f'PDF for interval -0.5 to 0.5 is {pdf(-.5,.5,x1,y1)}')
    print('')

    x2, y2 = normal.normal_distribution(5000, -20.0, 20.0, 0.0, 5.0)
    print(f'PDF for Normal Distribution with mu=0.0, sigma=5.0')
    print(f'PDF for interval -1   to 1   is {pdf(-1,1,x2,y2)}')
    print(f'PDF for interval -2   to 2   is {pdf(-2,2,x2,y2)}')
    print(f'PDF for interval -0.5 to 0.5 is {pdf(-.5,.5,x2,y2)}')
    print('')

    x3, y3 = normal.normal_distribution(5000, -20.0, 20.0, -5.0, 3.0)
    print(f'PDF for Normal Distribution with mu=-5.0, sigma=3.0')
    print(f'PDF for -5.0   is {pdf(-5,-5,x3,y3)}')
    print(f'PDF for  5.0   is {pdf( 5, 5,x3,y3)}')
    print(f'PDF for  0.0   is {pdf( 0.001, 0.001,x3,y3)}')
    print('')

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='mu= 0.0,sigma=1.0')
    ax.plot(x2,y2, label='mu= 0.0,sigma=5.0')
    ax.plot(x3,y3, label='mu=-5.0,sigma=3.0')
    ax.set_xlim(left=-10,right=10)
    ax.legend()
    ax.set_title("Normal Distributions")
    plt.grid()
    plt.show()
