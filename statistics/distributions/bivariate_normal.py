'''
PATREG - A Pattern Recognition Tool Box

bivariate_normal.py
Constructs a bivariate normal distribution given vector mean and an lxl covariance
matrix
'''
from math import *
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bivariate_normal(npoints: int, xmin: float, xmax: float, ymin: float, ymax: float,
                        mu: np.array, sigma: np.matrix) -> (np.array, np.array, np.array):

    # Set the number of variants.  This is a bivariant so it is 2
    l = 2

    # Initialize output matrix and output vector
    x   = np.linspace(xmin, xmax, npoints)
    y   = np.linspace(ymin, ymax, npoints)
    pdf = np.zeros(shape=(npoints,npoints))
    pxw1= np.zeros(npoints)
    pyw1= np.zeros(npoints)

    # Get the determinant of sigma
    det_sigma = det(sigma)

    # Get the inverse of the covariance matrix
    inv_sigma = inv(sigma)

    a = 1/((2*pi)**(l/2)*det_sigma**0.5)

    # Calculate PDF for x variate
    for i in range(npoints):
        x_mu_T = -0.5*np.transpose(x[i]-mu)
        x_mu   = x[i]-mu
        dist   = x_mu_T.dot(inv_sigma)
        dist   = dist.dot(x_mu)
    
        pxw1[i] = a*exp(dist)

    # Calculate PDF for y variate
    for i in range(npoints):
        x_mu_T = -0.5*np.transpose(y[i]-mu)
        x_mu   = y[i]-mu
        dist   = x_mu_T.dot(inv_sigma)
        dist   = dist.dot(x_mu)
    
        pyw1[i] = a*exp(dist)

    # Calculate the bivariate normal joint density
    pdf = np.outer(pxw1,pyw1)

    return (pdf, x, y)

if __name__=='__main__':

    # Set the mean vector and covariance matrix
    mu    = np.zeros(2)
    mu[0] = 0
    mu[1] = 0
    sigma = np.matrix('15 6;0 3')

    # Set limits
    xmin = -5
    xmax =  5
    ymin = -5
    ymax =  5

    # Use enough points to accurately model the curves
    npoints = 5000

    pdf,x,y = bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu, sigma)
    X,Y = np.meshgrid(x,y)

    fig,ax = plt.subplots()  
    cs     = ax.contourf(X,Y,pdf, cmap='jet')
    cbar   =fig.colorbar(cs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')



    plt.show()