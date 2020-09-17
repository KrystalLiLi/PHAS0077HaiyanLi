#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from scipy import constants as C
from matplotlib import pyplot as plt


# In[6]:


# This is a class to create a Langmuir adsorption model.
class LangmuirAdsorptionModel():

    def __init__(self, T, P):

        self.T = T

        self.P = P

        # Boltzmann's constant
        self.k_B = 8.6173 * 10 ** -5

        # 1 amu = 1.0364 * 10**-28 eV * S^2 * A**-2
        self.mass_conversion = 1.0364 * 10**(-28)
        # 1 bar = 6.2415 * 10 **(-7) eV * A**-3
        self.p_conversion = 6.2415 * 10**(-7)

        # Planck's constant
        self.h = 4.1357 * 10 ** -15

        self.I_X_gas = 9.093 * 10**-28

        self.A_site = 2.57

        self.mass = 28

        self.sigma = 1

        # vibrational frequency of CO adsorbed on bridge sites.
        self.wavenumbers = [1848, 399, 387, 374, 197, 51]

        # vibrational frequency of CO(g)
        self.wavenumber = 2127

        # binding energy of CO in this case is set to be 1.2 eV
        self.E_a_des = 1.2

    def k_ads_pre_expo(self):

        numerator = self.A_site

        square = 2 * math.pi * self.mass * self.mass_conversion * self.k_B * self.T

        square_root = square ** 0.5

        ads_pre_expo = numerator / square_root * self.p_conversion

        return ads_pre_expo

    # Vibrational partition function for CO ideal gas molecules.
    def vib_X_gas(self):

        # “Converting” wavenumbers (cm-1) to frequencies (s-1) requires
        # multiplying by the speed of light, c.
        exponent_numerator1 = - self.h * self.wavenumber * C.c
        exponent_numerator2 = 2 * self.k_B * self.T

        exponent_numerator = exponent_numerator1 / exponent_numerator2

        exponent_denominator = 2 * exponent_numerator

        q_vib_X_gas = math.exp(exponent_numerator) / \
            (1 - math.exp(exponent_denominator))

        return q_vib_X_gas

    # Rotational partition function for a linear molecule.
    def rot_X_gas(self):

        reduced_Plancks_constant = self.h / (2 * math.pi)

        theta_rot_X_gas = reduced_Plancks_constant**2 / \
            (2 * self.I_X_gas * self.k_B)

        denominator = self.sigma * theta_rot_X_gas

        q_rot_X_gas = self.T / denominator

        return q_rot_X_gas

    def trans2D_X_gas(self):

        numerator1 = 2 * math.pi * self.mass * self.mass_conversion
        numerator2 = self.k_B * self.T * self.A_site

        numerator = numerator1 * numerator2

        denominator = self.h**2

        q_trans2D_X_gas = numerator / denominator

        return q_trans2D_X_gas

    # Vibrational partition function for adsorbed CO.
    def vib_X(self):

        vib_X = []

        for i in range(len(self.wavenumbers)):

            wavenumber_ = self.wavenumbers[i]

            exponent_numerator1 = - self.h * wavenumber_ * C.c
            exponent_numerator2 = 2 * self.k_B * self.T

            exponent_numerator = exponent_numerator1 / exponent_numerator2

            exponent_denominator = 2 * exponent_numerator

            vib_X_i = math.exp(exponent_numerator) / \
                (1 - math.exp(exponent_denominator))

            vib_X.append(vib_X_i)

        q_vib_X = np.sum(vib_X)

        return q_vib_X

    def k_des(self):

        q_vib_X_gas = self.vib_X_gas()

        q_rot_X_gas = self.rot_X_gas()

        q_trans2D_X_gas = self.trans2D_X_gas()

        q_vib_X = self.vib_X()

        partition_function = q_vib_X_gas * q_rot_X_gas * q_trans2D_X_gas / q_vib_X

        des_pre_expo = self.k_B * self.T * partition_function / self.h

        denominator = self.k_B * self.T

        des_expo = np.exp(- self.E_a_des / denominator)

        k_des_ = des_pre_expo * des_expo

        return k_des_

    def coverage_calculation(self):

        ads_pre_expo = self.k_ads_pre_expo()

        constant_p = ads_pre_expo * self.P / self.k_des()

        coverage = constant_p / (1 + constant_p)

        return coverage
