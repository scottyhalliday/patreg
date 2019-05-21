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
                    lam_12: float   , lam_21: float) -> (int, float, float):
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

    px = px_pdf1*pw1 + px_pdf2*pw2

    # Calculate the probability of x in class w1 and x in class w2
    pw1x = px_pdf1*pw1/px
    pw2x = px_pdf2*pw2/px

    # Determine where the two classes intersect
    idx = np.argwhere(np.diff(np.sign(pxw1_y - pxw2_y))).flatten()

    # If multiple indicies intersect, find the index with the highest probability
    # and set that as our xo boundary
    xidx = -1
    xmax = -1
    for i in idx:
        if pxw1_y[i] > xmax:
            xidx = i
            xmax = pxw1_y[i]

    xo  = pxw1_x[xidx]

    # Calculate the probability of commiting a decision error
    Pe1 = 0
    Pe2 = 0

    for i in range(pxw2_x.size):
        if pxw2_x[i] > xo:
            break
        if i == 0:
            Pe1 = pxw2_y[i]
        else:
            Pe1 += (pxw2_x[i]-pxw2_x[i-1])*pxw2_y[i]

    for i in range(pxw1_x.size):
        if pxw2_x[i] < xo:
            continue
        if Pe2 == 0:
            Pe2 = pxw1_y[i]
        else:
            Pe2 += (pxw1_x[i]-pxw1_x[i-1])*pxw1_y[i]

    Pe = Pe1/2.0 + Pe2/2.0

    if pw1x > pw2x:
        return 1, pw1x, pw2x, Pe
    else:
        return 2, pw1x, pw2x, Pe
