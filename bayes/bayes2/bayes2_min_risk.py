'''
PATREG - A Pattern Recognition Tool Box

bayes2_min_risk.py
Models a two-class classifier based on Bayes Decision Theory with minimization
of risk with respect to probability of error of misclassification.

This builds on the simple Bayes two class system making a decision based not
just on the Bayes principle alone but also includes penalty terms as part of 
the decision criteria.  This will help alleviate cases where the probability
of error with respect to a decision are high.
'''

from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
from patreg.statistics.distributions import normal, gumbel, cauchy
from patreg.statistics.pdf import pdf

def bayes2_min_risk(x: float, pw1: float, pw2: float,
                    pxw1_x: np.array, pxw1_y: np.array,
                    pxw2_x: np.array, pxw2_y: np.array,
                    L: np.matrix) -> (int, float, float):
    """
    Calculate the probability that random variable x is in class w1 or w2.  This
    also will calculate the probability of error on the class selection.

    :param float x: Random variable value (r.v)
    :param float pw1: The priori probability that a value is class w1
    :param float pw2: The priori probability that a value is class w2
    :param np.array pxw1_x: The x-component of the conditional probability
    density function p(x|w1).  This is the likelihood that r.v. x belongs w1
    :param np.array pxw1_y: The y-component of the conditional probability
    density function p(x|w1).  This is the likelihood that r.v. x belongs w1
    :param np.array pxw2_x: The x-component of the conditional probability
    density function p(x|w2).  This is the likelihood that r.v. x belongs w2
    :param np.array pxw2_y: The y-component of the conditional probability
    :param float L: The loss/risk matrix used to make classification decision
    density function p(x|w2).  This is the likelihood that r.v. x belongs w2
    :rtype: float
    :raises AssertionError: If the x and y array lengths are not the same for
    each pdf.
    """

    # Make sure array sizes are consistent
    assert(pxw1_x.size == pxw1_y.size)
    assert(pxw2_x.size == pxw2_y.size)

    # Make sure the priori probabilities add up to one
    assert(abs(pw1+pw2-1.0)<1e-6)

    # Calculate the probability of the random value x.  Offset by 0.5 so that
    # the pdf (which is an area under the curve) only multiplies the
    # probability by 1.0.
    px_pdf1 = pdf(x, x, pxw1_x, pxw1_y)
    px_pdf2 = pdf(x, x, pxw2_x, pxw2_y)

    # Calculate the likelihoods for each class
    l1 = L[0,0]*px_pdf1*pw1 + L[1,0]*px_pdf2*pw2
    l2 = L[0,1]*px_pdf1*pw1 + L[1,1]*px_pdf2*pw2

    if l1 < l2:
        return 1, l1, l2
    else:
        return 2, l1, l2

if __name__=='__main__':

    L = np.matrix('0 0.5; 1.0 0')

    xw1,yw1 = normal.normal_distribution(5000, -20.0, 20.0, -5.0, 3.0)
    xw2,yw2 = normal.normal_distribution(5000, -20.0, 20.0,  5.0, 3.0)

    xclasses = []

    # A priori probabilities
    Pw1 = 0.5
    Pw2 = 0.5

    # List of random variales to classify
    xs = [-5, 5, -1, 1, 0, -10, 10]

    print(f'CLASSIFIERS FOR TWO NORMAL DISTRIBUTIONS')
    for x in xs:
        xclass,l1,l2 = bayes2_min_risk(x, Pw1, Pw2, xw1,yw1, xw2,yw2, L)
        print(f'Random variable {x:6.2f} belongs to class {xclass}, l1={l1:.3f}, l2={l2:.3f}')
        xclasses.append((x, xclass,l1,l2))

    fig,ax = plt.subplots(1,2)
    ax[0].plot(xw1,yw1, label='p(x|w1) - Class 1')
    ax[0].plot(xw2,yw2, label='p(x|w2) - Class 2')
    for (x,xclass,l1,l2) in xclasses:
        ax[0].plot(x,0,'o', label=f'{x} in class {xclass}')
    ax[0].set_xlim(left=-20,right=20)
    ax[0].legend()
    ax[0].set_title(f'Region Classifiers with Risk Minimization\nWith a priori probabilities of Pw1={Pw1} and Pw2={Pw2}')
    ax[0].set_ylabel('p(x|w)')
    ax[0].set_xlabel('x')
    ax[0].grid()

    print('\n\nCLASSIFIERS FOR GUMBEL AND CAUCHY DISTRIBUTIONS')

    xw3,yw3 = gumbel.gumbel_distribution(5000, -20.0, 20.0, 0.5, 5.0)
    xw4,yw4 = cauchy.cauchy_distribution(5000, -20.0, 20.0, 7.0, 1.0)

    xclasses = []

    # A priori probabilities
    Pw1 = 0.5
    Pw2 = 0.5

    # List of random variales to classify
    xs = [-5, 5, -1, 1, 0, -10, 10]

    for x in xs:
        xclass,l1,l2 = bayes2_min_risk(x, Pw1, Pw2, xw3,yw3, xw4,yw4, L)
        print(f'Random variable {x:6.2f} belongs to class {xclass}, l1={l1:.3f}, l2={l2:.3f}')
        xclasses.append((x, xclass,l1,l2))

    ax[1].plot(xw3,yw3, label='p(x|w1) - Class 1')
    ax[1].plot(xw4,yw4, label='p(x|w2) - Class 2')
    for (x,xclass,l1,l2) in xclasses:
        ax[1].plot(x,0,'o', label=f'{x} in class {xclass}')
    ax[1].set_xlim(left=-20,right=20)
    ax[1].legend()
    ax[1].set_title(f'Region Classifiers with Risk Minimization\nWith a priori probabilities of Pw1={Pw1} and Pw2={Pw2}')
    ax[1].set_ylabel('p(x|w)')
    ax[1].set_xlabel('x')
    ax[1].grid()

    #plt.grid()
    plt.show()
