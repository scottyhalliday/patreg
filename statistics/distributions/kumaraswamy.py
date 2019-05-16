'''
PATREG - A Pattern Recognition Tool Box

kumaraswamy.py
A family of continuous probability distributions defined on the interval of (0,1).  It is similiar
to the beta distribution but much simpler to calculate
'''
from cmath import *
import numpy as np
import matplotlib.pyplot as plt

def kumaraswamy_distribution(num_points: int,    # Number of points to evaluate
                             a         : float,  # Non-negative shape parameter
                             b         : float   # Non-negative shape parameter
                            ) -> (np.array, np.array):

    assert( a > 0 )
    assert( b > 0 )
    
    kumar = np.zeros(num_points)

    x=np.linspace(0, 1, num_points)

    for i in range(1,num_points):
        kumar[i] = a*b*(x[i]**(a-1)) * ((1-x[i]**a)**(b-1))

    return x,kumar

if __name__=='__main__':
    

    x1,y1 = kumaraswamy_distribution(100, 0.5, 0.5)
    x2,y2 = kumaraswamy_distribution(1000, 5.0, 1.0)
    x3,y3 = kumaraswamy_distribution(1000, 1.0, 3.0)
    x4,y4 = kumaraswamy_distribution(1000, 2.0, 2.0)
    x5,y5 = kumaraswamy_distribution(1000, 2.0, 5.0)

    fig,ax = plt.subplots()
    ax.plot(x1,y1, label='a=0.5,b=0.5')
    ax.plot(x2,y2, label='a=5.0,b=1.0')
    ax.plot(x3,y3, label='a=1.0,b=3.0')
    ax.plot(x4,y4, label='a=2.0,b=2.0')
    ax.plot(x5,y5, label='a=2.0,b=5.0')

    ax.set_xlim(left=0,right=1)
    ax.set_ylim(top=3.0, bottom=0)
    ax.legend()
    ax.set_title("Kumaraswamy Distributions")
    plt.grid()
    plt.show()
