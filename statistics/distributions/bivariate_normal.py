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
    xy  = np.zeros(2)
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
        for j in range(npoints):
            xy[:]  = x[i],y[j]
            x_mu_T = np.transpose(xy-mu)
            x_mu   = xy-mu
            dist   = x_mu_T.dot(inv_sigma)
            dist   = dist.dot(x_mu)
    
            pdf[i][j] = a*exp(-0.5*dist)

    return (pdf, x, y)

if __name__=='__main__':

    # Set the mean vector and covariance matrix
    mu    = np.zeros(2)
    sigma = np.matrix('3 0;0 3')

    # Set limits
    xmin = -10
    xmax =  10
    ymin = -10
    ymax =  10

    # Use enough points to accurately model the curves
    npoints = 100

    pdf,x,y = bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu, sigma)
    X,Y = np.meshgrid(x,y)

    fig,ax = plt.subplots()  
    cs = plt.contourf(X,Y,pdf,levels=20, cmap='jet')
    
    # Add a color bar which maps values to colors.
    cbar = plt.colorbar(cs)

    plt.show()