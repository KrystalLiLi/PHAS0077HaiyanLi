#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import math
from scipy import constants as C
from matplotlib import pyplot as plt
from LangmuirAdsorptionModel import LangmuirAdsorptionModel
from OnLatticeKMC import KMC_lattice


# In[13]:


def Comparison_KMC_Langmuir(T, P, height, width):

    LangmuirModel = LangmuirAdsorptionModel(T, P)

    Langmuir_coverage = LangmuirModel.coverage_calculation()

    print('The coverage calculated from the Langmuir isotherm is {0}.'.format(
        Langmuir_coverage))

    k_ads = LangmuirModel.k_ads_pre_expo()

    k_des_ = LangmuirModel.k_des()

    KMC_model = KMC_lattice(height, width, k_ads, P, k_des_)

    KMC_model.plot()


# In[15]:
Comparison_KMC_Langmuir(580, 0.004, 10, 10)


# In[ ]:


# In[ ]:
