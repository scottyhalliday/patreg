'''
PATREG - A Pattern Recognition Tool Box

bayes2.py
Models a two class classifier based on Bayes Decision Theory.  This is a
two-class case only.

This assumes that there are two classes w1 and w2 to
which the pattern belongs.  Given the priori probabilities of w1 and w2 along
with the class-conditional probability density functions p(x|w1) and p(x|w2).
The PDF p(x|wi) is sometimes referred to as the likelihood function of wi with
respect to random variable x.  To summarize the following items are needed
to determine which class a random variable x will belong to
    - Prior probabilities for w1 and w2, that is P(w1) and P(w2).  This is the
      prior knowledge of how likely it is to get w1 or w2
    - Conditional probability densities p(x|w1) and p(x|w2).  That is how
      frequentlky we will measure a pattern with feature value x given that
      the pattern belongs to class wi
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Import this packages modules
from patreg.statistics.distributions import normal
from patreg.statistics.pdf import pdf

def bayes2(x: float, pw1: float, pw2: float,
           pxw1_x: np.array, pxw1_y: np.array,
           pxw2_x: np.array, pxw2_y: np.array) -> (int, float, float, float):
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
    xo  = pxw1_x[idx]

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

if __name__=='__main__':
    xw1,yw1 = normal.normal_distribution(5000, -20.0, 20.0, -5.0, 3.0)
    xw2,yw2 = normal.normal_distribution(5000, -20.0, 20.0,  5.0, 3.0)

    xclasses = []

    xclass,px1,px2,Pe = bayes2(-5, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable  -5.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append(( -5.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2( 5, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable   5.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append((  5.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2(-1, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable  -1.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append(( -1.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2( 1, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable  -1.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append((  1.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2( 0, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable   0.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append((  0.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2(-10, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable -10.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append(( -10.0, xclass,px1,px2,Pe))

    xclass,px1,px2,Pe = bayes2(10, 0.5, 0.5, xw1,yw1, xw2,yw2)
    print(f'Random variable  10.0 belongs to class {xclass}, pxw1={px1:.3f}, pxw2={px2:.3f}, Pe={Pe:.3f}')
    xclasses.append(( 10.0, xclass,px1,px2,Pe))

    fig,ax = plt.subplots()
    ax.plot(xw1,yw1, label='p(x|w1) - Class 1')
    ax.plot(xw2,yw2, label='p(x|w2) - Class 2')
    for (x,xclass,px1,px2,Pe) in xclasses:
        ax.plot(x,0,'o', label=f'{x} in class {xclass}')
    ax.set_xlim(left=-20,right=20)
    ax.legend()
    ax.set_title(f'Region Classifiers with Probability of Error {100*Pe:.2f}%\nWith a priori probabilities of Pw1=0.5 and Pw2=0.5')
    ax.set_ylabel('p(x|w)')
    ax.set_xlabel('x')
    plt.grid()
    plt.show()
