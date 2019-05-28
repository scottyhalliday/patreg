'''
PATREG - A Pattern Recognition Tool Box

bayes_min_distance.py
Models two classes made up of bivariate normal distribution with the
SAME covariance matrix.  This program will compute the minimum distance
to a class using the Mahalanobis distance
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import *

# Import this packages modules
from patreg.statistics.distributions import bivariate_normal

def bayes_min_distance(x1: float, x2: float, mu1: np.array, mu2: np.array, sigma: np.matrix ) -> (int, float, float):
    """
    Determines which class a set of random variables belongs to in a bivariate
    normal distribution.

    :param float x1: Random variable value (r.v) x1 to make up the 2D point
    (x1,x2)
    :param float x2: Random variable value (r.v) x2 to make up the 2D point
    (x1,x2)
    :param np.matrix pdfw1: The bivariate normal distribution for class w1
    :param np.matrix pdfw2: The bivariate normal distribution for class w2
    :param np.array x: The x coordinates which make up the bivariate normal
    distribution
    :param np.array y: The y coordinates which make up the bivariate normal
    distribution
    :param np.array mu1: The means for x1,x2 variable set for bivariate normal
    distribution for class 1
    :param np.array mu2: The means for x1,x2 variable set for bivariate normal
    distribution for class 2
    :param np.matrix sigma1: The covariance matrix for x1,x2 variable set for 
    bivariate normal distribution for BOTH classes
    :rtype: float
    :raises AssertionError: If the mu arrays are not the same size.
    """
    
    # Make sure sigma's and mu's agree in size
    assert(mu1.size    == mu2.size)

    # Set up the matrices
    X = np.matrix('0;0',dtype=float)
    X[0,0] = x1
    X[1,0] = x2

    mu1M = np.matrix('0;0',dtype=float)
    mu1M[0,0] = mu1[0]
    mu1M[1,0] = mu1[1]

    mu2M = np.matrix('0;0',dtype=float)
    mu2M[0,0] = mu2[0]
    mu2M[1,0] = mu2[1]    

    # Inverse sigma
    inv_sigma = inv(sigma)

    # Calculate distances, smallest distance is the class to which the vector belongs
    dm1 = np.transpose(X-mu1M)*inv_sigma*(X-mu1M)
    dm2 = np.transpose(X-mu2M)*inv_sigma*(X-mu2M)

    if dm1 < dm2:
        return 1,dm1[0,0],dm2[0,0]
    else:
        return 2,dm1[0,0],dm2[0,0]

if __name__=='__main__':

    # Vector to classify
    v = np.zeros(2)
    v[0] = 1.0
    v[1] = 2.2

    # Covariance matrix shared by both distributions
    sigma = np.matrix('1.1 0.3;0.3 1.9')

    # Mean vectors
    mu1   = np.zeros(2)
    mu2   = np.ones(2)*3.0

    xclass,dm1,dm2=bayes_min_distance(v[0],v[1],mu1,mu2,sigma)

    print(f'Vector {v[0]}, {v[1]} belongs to class {xclass}, dm1={dm1:9.3f}, dm2={dm2:.3f}')

    # Setup the number of points and boundaries of distributions
    npoints =  100
    xmax    =  8
    xmin    = -8
    ymax    =  8
    ymin    = -8

    # Calculate the distributions
    w1,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu1, sigma)
    w2,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu2, sigma)

    # Plot distributions
    X,Y = np.meshgrid(x,y)

    fig,ax = plt.subplots()  
    cs1 = ax.contour(X,Y,w1,levels=15, cmap='jet')
    cs2 = ax.contour(X,Y,w2,levels=15, cmap='jet')

    ax.clabel(cs1, inline=1, fontsize=6)
    ax.clabel(cs2, inline=1, fontsize=6)

    ax.plot(v[0],v[1], 'ro')

    ax.set_title('Bivariate Distribution Min Distance Bayesian Classifier ')
    ax.set_ylabel('x1')
    ax.set_xlabel('x2')
    ax.grid()

    plt.show()