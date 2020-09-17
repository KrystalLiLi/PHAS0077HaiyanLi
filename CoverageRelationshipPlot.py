#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from scipy import constants as C
from matplotlib import pyplot as plt
from LangmuirAdsorptionModel import LangmuirAdsorptionModel


# In[5]:


# This function gives the plot showing the relationship
# between the coverage and the partial pressure under different
# temperatures of CO.
def CoveragePlot():

    range_T = np.arange(500, 700, 20).tolist()

    range_P = np.arange(0.0001, 0.01, 0.001).tolist()

    coverage_T = []

    for i in range_T:

        # Suppose temperature T is a constant, investigate how the coverage changes
        # as the partial pressure increases.
        coverage_P = []

        for j in range_P:

            Model = LangmuirAdsorptionModel(i, j)

            theta = Model.coverage_calculation()

            coverage_P.append(theta)

        coverage_T.append(coverage_P)

    plt.figure()

    plt.plot(range_P, coverage_T[0], label='T = 500 K')
    plt.plot(range_P, coverage_T[1], label='T = 520 K')

    plt.plot(range_P, coverage_T[2], label='T = 540 K')
    plt.plot(range_P, coverage_T[3], label='T = 560 K')
    plt.plot(range_P, coverage_T[4], label='T = 580 K')
    plt.plot(range_P, coverage_T[5], label='T = 600 K')
    plt.plot(range_P, coverage_T[6], label='T = 620 K')
    plt.plot(range_P, coverage_T[7], label='T = 640 K')

    plt.plot(range_P, coverage_T[8], label='T = 660 K')
    plt.plot(range_P, coverage_T[9], label='T = 680 K')

    plt.xlabel('Pressure(bar)')
    plt.ylabel('Coverage')
    plt.xlim((0.0, 0.0125))
    plt.legend(loc='right')
    plt.title(
        'Relationship between coverage and pressure and temperature of CO',
        pad=20)
    plt.savefig(
        'Relationship between coverage and pressure and temperature of CO')
    plt.show()


# In[6]:

CoveragePlot()


# In[ ]:
