'''
PATREG - A Pattern Recognition Tool Box

bayes_bivariate.py
Models two class with Bivariate Normal distribution and determines
which class a given random variable is classified based on Bayes
Decision Theory
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import *

# Import this packages modules
from patreg.statistics.distributions import bivariate_normal

def bayes_bivariate(x1: float        , x2: float        , pdfw1: np.matrix, pdfw2: np.matrix,
                    x: np.array      , y: np.array      , mu1: np.array   , mu2: np.array   ,
                    sigma1: np.matrix, sigma2: np.matrix, Pw1: float      , Pw2: float       ) -> (int, float, float):
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
    bivariate normal distribution for class 1
    :param np.matrix sigma2: The covariance matrix for x1,x2 variable set for 
    bivariate normal distribution for class 2
    :param float Pw1: The priori probability that a value is class w1
    :param float Pw2: The priori probability that a value is class w2
    :rtype: float
    :raises AssertionError: If the mu arrays are not the same size or the sigma
    covariance matrices are not the same size.
    """
    
    # Make sure sigma's and mu's agree in size
    assert(mu1.size    == mu2.size)
    assert(sigma2.size == sigma2.size)

    # Set up the random variable matrix
    X = np.zeros((2,1))
    X[0,0] = x1
    X[1,0] = x2

    # Transpose X
    Xt = np.transpose(X)

    # Convert mu's to matricies
    mu1M = np.matrix('0;0')
    mu2M = np.matrix('0;0')

    mu1M[0,0] = mu1[0]
    mu1M[1,0] = mu1[1]
    mu2M[0,0] = mu2[0]
    mu2M[1,0] = mu2[1]

    # Transpose mu's
    mu1T = np.transpose(mu1M)
    mu2T = np.transpose(mu2M)

    # Inverse sigma's
    inv_sigma1 = inv(sigma1)
    inv_sigma2 = inv(sigma2)

    # Determinant of sigma's
    det_sigma1 = det(sigma1)
    det_sigma2 = det(sigma2)

    # Because of the exponential form of the involved densities, it is
    # preferable to work with the following discriminant functions which
    # involve the monotonic logarithmic function ln(.):
    #    g1(x) = ln(p(x|w1)) + ln(Pw1)
    #    g2(x) = ln(p(x|w2)) + ln(Pw2)
    # 
    # While these equations can be solved with the following code:
    # (Uncomment the code between the dashes to see)
    #
    # -------------------------------------------------------------------------
    # pxw1 = mv_probability(x1,x2,w1,x,y)
    # pxw2 = mv_probability(x1,x2,w2,x,y)
    #
    # g1x  = np.log(pxw1) + np.log(Pw1)
    # g2x  = np.log(pxw2) + np.log(Pw2)
    # -------------------------------------------------------------------------
    #
    # There will be error with respect to this calculation due to the
    # discretized distributions.  The more points which make up the
    # distribution will reduce the error but at a cost for significant
    # computation time.
    # 
    # A better solution is to expand out the equation shown above and solve it
    # in terms of x1,x2,mu1,m2,sigma1 and sigma2 keeping in mind how the bivariate
    # distribution is defined.  Its derivation will not be done here and one
    # can refer to any text book on the subject to better understand it

    # Calculate the constants
    c1 = np.log(2*pi) - 0.5*np.log(det_sigma1)
    c2 = np.log(2*pi) - 0.5*np.log(det_sigma2)

    # Solve for each class' descriminant value 
    t11 = -0.5*Xt*inv_sigma1*X + 0.5*Xt*inv_sigma1*mu1M - 0.5*mu1T*inv_sigma1*mu1M
    t12 = 0.5*mu1T*inv_sigma1*X + np.log(Pw1) + c1
    g1X = t11 + t12

    t21 = -0.5*Xt*inv_sigma2*X + 0.5*Xt*inv_sigma2*mu2M - 0.5*mu2T*inv_sigma2*mu2M
    t22 = 0.5*mu2T*inv_sigma2*X + np.log(Pw2) + c2
    g2X = t22 + t21
    
    # Determine which class based on the descriminant values
    if g1X > g2X:
        return 1,g1X[0,0],g2X[0,0]
    else:
        return 2,g1X[0,0],g2X[0,0]


def mv_probability(x1: float, x2: float, w1: np.array, x: np.array, y:np.array) -> float:
    '''
    Calculates the probability of a multivariate normal distribution given the 
    distribution over x-y grid and a unique point in 2D space, x1 and x2
    '''
    assert(x.size == y.size)

    # Since this is a discrete distribution find the values in x and y which are
    # closest to the desired points x1 and x2.
    x1x = x1-x
    x2x = x2-y

    # Get the indicies for these points and return the probability from w1
    idx1 = (np.abs(x1x)).argmin()
    idx2 = (np.abs(x2x)).argmin()

    return w1[idx1,idx2]


if __name__=='__main__':

    npoints =  100
    xmax    =  8
    xmin    = -8
    ymax    =  8
    ymin    = -8

    Pw1     = 0.5
    Pw2     = 0.5

    mu1     = np.zeros(2)
    mu2     = np.zeros(2)
    mu2[0]  = 4
    mu3     = np.zeros(2)
    mu4     = np.zeros(2)
    mu4[0]  = 3.2

    sigma1  = np.matrix('0.30 0.0;0.0 0.35')
    sigma2  = np.matrix('1.20 0.0;0.0 1.85')
    sigma3  = np.matrix('0.10 0.0;0.0 0.75')
    sigma4  = np.matrix('0.75 0.0;0.0 0.10')

    w1,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu1, sigma1)
    w2,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu2, sigma2)
    w3,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu3, sigma3)
    w4,x,y = bivariate_normal.bivariate_normal(npoints, xmin,xmax, ymin,ymax, mu4, sigma4)

    print_summary1 = []
    print_summary2 = []
    x1s = [-6, -4, -2, 0, 2, 4, 6]
    x2s = [-6, -4, -2, 0, 2, 4, 6]

    # You can use linspace to paint ALOT more points to really see the class
    # distribution.  It is slower but visually makes sense.  
    #x1s = np.linspace(-6, 6, 25)
    #x2s = np.linspace(-6, 6, 25)

    print("Bayes Classification Set 1\n")
    for x1 in x1s:
        for x2 in x2s:
            xclass, g1x, g2x = bayes3(x1,x2,w1,w2,x,y,mu1,mu2,sigma1,sigma2,Pw1,Pw2)
            print(f'x1={x1:6.2f}, x2={x2:6.2f} is in class {xclass}.  g1x={g1x:9.3f}, g2x={g2x:9.3f}')
            print_summary1.append((xclass, x1, x2))

    print("Bayes Classification Set 2\n")
    for x1 in x1s:
        for x2 in x2s:
            xclass, g1x, g2x = bayes3(x1,x2,w1,w2,x,y,mu3,mu4,sigma3,sigma4,Pw1,Pw2)
            print(f'x1={x1:6.2f}, x2={x2:6.2f} is in class {xclass}.  g1x={g1x:9.3f}, g2x={g2x:9.3f}')
            print_summary2.append((xclass, x1, x2))

    # Plot the 2 classes
    X,Y = np.meshgrid(x,y)

    fig,ax = plt.subplots(1,2)  
    cs1 = ax[0].contour(X,Y,w1,levels=15, cmap='jet')
    cs2 = ax[0].contour(X,Y,w2,levels=15, cmap='jet')
    cs3 = ax[1].contour(X,Y,w3,levels=15, cmap='jet')
    cs4 = ax[1].contour(X,Y,w4,levels=15, cmap='jet')

    ax[0].clabel(cs1, inline=1, fontsize=6)
    ax[0].clabel(cs2, inline=1, fontsize=6)
    ax[1].clabel(cs3, inline=1, fontsize=6)
    ax[1].clabel(cs4, inline=1, fontsize=6)

    # Plot the points
    label1_set = False
    label2_set = False
    for pset in print_summary1:
        xclass = pset[0]
        x1     = pset[1]
        x2     = pset[2]
        if xclass == 1:
            if not label1_set:
                ax[0].plot(x1,x2,'ro', label='Class 1')
                label1_set = True
            else:
                ax[0].plot(x1,x2,'ro')
        else:

            if not label2_set:
                ax[0].plot(x1,x2,'bo', label='Class 2')
                label2_set = True
            else:
                ax[0].plot(x1,x2,'bo')

    label1_set = False
    label2_set = False
    for pset in print_summary2:
        xclass = pset[0]
        x1     = pset[1]
        x2     = pset[2]
        if xclass == 1:
            if not label1_set:
                ax[1].plot(x1,x2,'ro', label='Class 1')
                label1_set = True
            else:
                ax[1].plot(x1,x2,'ro')
        else:

            if not label2_set:
                ax[1].plot(x1,x2,'bo', label='Class 2')
                label2_set = True
            else:
                ax[1].plot(x1,x2,'bo')

    ax[0].set_title('Bivariate Distribution 1 Bayesian Classifier ')
    ax[0].set_ylabel('x1')
    ax[0].set_xlabel('x2')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].set_title('Bivariate Distribution 2 Bayesian Classifier ')
    ax[1].set_ylabel('x1')
    ax[1].set_xlabel('x2')
    ax[1].legend()
    ax[1].grid()

    plt.show()