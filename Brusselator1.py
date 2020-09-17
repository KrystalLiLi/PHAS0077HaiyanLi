#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
from matplotlib import pyplot as plt


# In[34]:


# In this class, numbers of X1 and X2 are fixed.
# The parameter "type_index" determines if it is direct method or the first-reaction method.
# If it is 1, then direct method is used. If it is 2, then first-reaction
# is used.
class Brusselator_1():

    def __init__(self, t_max, c1X1, c2X2, c3, c4, Y1, Y2, type_index):

        # Now t_max is not the KMC simulation time limit but the loop times
        # limit.
        self.t_max = t_max

        self.c1X1 = c1X1

        self.c2X2 = c2X2

        self.c3 = c3

        self.c4 = c4

        self.Y1 = Y1

        self.Y2 = Y2

        # alpha1 = h1 * c1 = X1 * c1 = c1X1
        self.alpha1 = self.c1X1

        # alpha2 = h2 * c2 = X2 * Y1 * c2 = c2X2 * Y1
        self.alpha2 = self.c2X2 * self.Y1

        # alpha3 = h3 * c3 = Y2 * Y1 * (Y1 -1) * 0.5 * c3
        self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3

        # alpha4 = h4 * c4 = Y1 * c4
        self.alpha4 = self.Y1 * self.c4

        self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4

        self.t = 0

        self.type_index = type_index

    def interval_time_direct_method(self):

        r1 = np.random.rand()

        tau = - 1.0 / self.alpha0 * np.log(1.0 - r1)

        return tau

    def event_selection_direct_method(self):

        r2 = np.random.rand()

        if 0 < r2 * self.alpha0 < self.alpha1:
            mu = 1
        elif self.alpha1 < r2 * self.alpha0 < self.alpha1 + self.alpha2:
            mu = 2
        elif self.alpha1 + self.alpha2 < r2 * self.alpha0 < self.alpha1 + self.alpha2 + self.alpha3:
            mu = 3
        elif self.alpha1 + self.alpha2 + self.alpha3 < r2 * self.alpha0 < self.alpha0:
            mu = 4
        return mu

    def KMC_direct_method(self):

        Y1_molecule = [self.Y1]
        Y2_molecule = [self.Y2]
        t_series = [self.t]

        for i in range(self.t_max):

            if self.alpha0 == 0:

                break

            self.tau = self.interval_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            elif self.mu == 3:
                self.Y1 += 1
                self.Y2 -= 1
            else:
                self.Y1 -= 1

            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha2 = self.c2X2 * self.Y1
            self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3
            self.alpha4 = self.Y1 * self.c4

            self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4

            self.t += self.tau
            t_series.append(self.t)

        return Y1_molecule, Y2_molecule, t_series

    def first_reaction_method(self):

        time_list = []

        for i in range(len(self.alpha_list)):

            r1 = np.random.rand()

            interval_time = - 1.0 / self.alpha_list[i] * np.log(1.0 - r1)

            time_list.append(interval_time)

        tau = min(time_list)

        mu = time_list.index(tau) + 1

        return tau, mu

    def KMC_first_reaction_method(self):

        Y1_molecule = [self.Y1]
        Y2_molecule = [self.Y2]
        t_series = [self.t]

        for i in range(self.t_max):

            self.alpha_list = [
                self.alpha1,
                self.alpha2,
                self.alpha3,
                self.alpha4]

            if self.alpha1 == 0 or self.alpha2 == 0 or self.alpha3 == 0 or self.alpha4 == 0:

                break

            self.tau, self.mu = self.first_reaction_method()

            if self.mu == 1:
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            elif self.mu == 3:
                self.Y1 += 1
                self.Y2 -= 1
            else:
                self.Y1 -= 1

            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha2 = self.c2X2 * self.Y1
            self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3
            self.alpha4 = self.Y1 * self.c4

            self.t += self.tau
            t_series.append(self.t)

        return Y1_molecule, Y2_molecule, t_series

    def plot(self):

        if self.type_index == 1:

            Y1_molecule, Y2_molecule, t_series = self.KMC_direct_method()

            graph_title = 'direct method'
            savefig_title = 'Brusselator1_direct method'
        else:
            Y1_molecule, Y2_molecule, t_series = self.KMC_first_reaction_method()
            graph_title = 'first-reaction method'
            savefig_title = 'Brusselator1_first-reaction method'

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(
            t_series,
            Y1_molecule,
            color='k',
            label='number of Y1 molecules')
        plt.xlabel('time')
        plt.ylabel('number of Y1 molecules')
        plt.legend()
        plt.text(0, 6000, 'a')

        plt.subplot(2, 2, 2)
        plt.plot(
            t_series,
            Y2_molecule,
            color='k',
            label='number of Y2 molecules')
        plt.xlabel('time')
        plt.ylabel('number of Y2 molecules')
        plt.legend()
        plt.text(0, 6000, 'b')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle(graph_title)

        plt.subplot(2, 2, 3)
        plt.plot(Y1_molecule, Y2_molecule, color='k')
        plt.xlabel('number of Y1 molecules')
        plt.ylabel('number of Y2 molecules')
        plt.text(6500, 6500, 'c')

        plt.savefig(savefig_title)
        plt.show()


# In[35]:


def main_direct_method():
    Brusselator1 = Brusselator_1(2000000, 5000, 50, 0.00005, 5, 1000, 2000, 1)

    Brusselator1.plot()


# In[36]:


main_direct_method()


# In[37]:


def main_first_reaction_method():
    Brusselator1 = Brusselator_1(2000000, 5000, 50, 0.00005, 5, 1000, 2000, 2)
    Brusselator1.plot()


# In[38]:

main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:
