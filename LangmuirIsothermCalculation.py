#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import math
from scipy import constants as C
from matplotlib import pyplot as plt
from LangmuirAdsorptionModel import LangmuirAdsorptionModel


# In[22]:


def LangmuirModelCalculation(T, P):

    Model = LangmuirAdsorptionModel(T, P)

    coverage = Model.coverage_calculation()

    print('The coverage of this Langmuir model is {0}'.format(coverage))


# In[23]:


LangmuirModelCalculation(580, 0.004)


# In[ ]:
