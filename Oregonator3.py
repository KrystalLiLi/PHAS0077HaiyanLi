#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from matplotlib import pyplot as plt


# In[25]:


class Oregonator_3():

    def __init__(self, t_max, X2, Y1, Y2, Y3, rho1, rho2, type_index):

        self.t_max = t_max  # The maximum KMC simulation time.

        self.X2 = X2

        self.Y1 = Y1

        self.Y2 = Y2

        self.Y3 = Y3

        self.rho1 = rho1

        self.rho2 = rho2

        self.c1X1 = self.rho1 / self.Y2

        self.c2 = self.rho2 / (self.Y1 * self.Y2)

        self.c3 = (self.rho1 + self.rho2) / (self.Y1 * self.X2)

        self.c4 = 2 * self.rho1 / (self.Y1**2)

        self.c5X3 = (self.rho1 + self.rho2) / self.Y3

        # alpha1 = h1 * c1 = X1 * Y2 * c1 = c1X1 * Y2
        self.alpha1 = self.c1X1 * self.Y2

        # alpha2 = h2 * c2 = Y1 * Y2 * c2
        self.alpha2 = self.Y1 * self.Y2 * self.c2

        # alpha3 = h3 * c3 = X2 * Y1 * c3
        self.alpha3 = self.c3 * self.X2 * self.Y1

        # alpha4 = h4 * c4 = 0.5 * Y1 * (Y1 - 1) * c4
        self.alpha4 = 0.5 * self.Y1 * (self.Y1 - 1) * self.c4

        # alpha5 = h5 * c5 = X3 * Y3 * c5 = c5X3 * Y3
        self.alpha5 = self.c5X3 * self.Y3

        self.alpha0 = self.alpha1 + self.alpha2 + \
            self.alpha3 + self.alpha4 + self.alpha5

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
        elif self.alpha1 + self.alpha2 + self.alpha3 < r2 * self.alpha0 < self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4:
            mu = 4

        elif self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4 < r2 * self.alpha0 < self.alpha0:
            mu = 5

        return mu

    def KMC_direct_method(self):

        Y1_molecule = [self.Y1]
        Y2_molecule = [self.Y2]
        Y3_molecule = [self.Y3]
        X2_molecule = [self.X2]
        t_series = [self.t]

        while self.t < self.t_max:

            if self.alpha0 == 0:

                break

            self.tau = self.interval_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:
                self.Y1 += 1
                self.Y2 -= 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 -= 1
            elif self.mu == 3:
                self.X2 -= 1
                self.Y1 += 1
                self.Y3 += 1
            elif self.mu == 4:
                self.Y1 -= 2
            else:
                self.Y3 -= 1
                self.Y2 += 1

            self.t += self.tau
            t_series.append(self.t)

            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)
            Y3_molecule.append(self.Y3)
            X2_molecule.append(self.X2)

            self.alpha1 = self.c1X1 * self.Y2
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.c3 * self.X2 * self.Y1
            self.alpha4 = 0.5 * self.Y1 * (self.Y1 - 1) * self.c4
            self.alpha5 = self.c5X3 * self.Y3

            self.alpha0 = self.alpha1 + self.alpha2 + \
                self.alpha3 + self.alpha4 + self.alpha5

        return Y1_molecule, Y2_molecule, Y3_molecule, X2_molecule, t_series

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
        Y3_molecule = [self.Y3]
        X2_molecule = [self.X2]
        t_series = [self.t]

        while self.t < self.t_max:

            self.alpha_list = [
                self.alpha1,
                self.alpha2,
                self.alpha3,
                self.alpha4,
                self.alpha5]

            if self.alpha1 == 0 or self.alpha2 == 0 or self.alpha3 == 0 or self.alpha4 == 0 or self.alpha5 == 0:

                break

            self.tau, self.mu = self.first_reaction_method()

            if self.mu == 1:
                self.Y1 += 1
                self.Y2 -= 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 -= 1
            elif self.mu == 3:
                self.X2 -= 1
                self.Y1 += 1
                self.Y3 += 1
            elif self.mu == 4:
                self.Y1 -= 2
            else:
                self.Y3 -= 1
                self.Y2 += 1

            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)
            Y3_molecule.append(self.Y3)
            X2_molecule.append(self.X2)

            self.alpha1 = self.c1X1 * self.Y2
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.c3 * self.X2 * self.Y1
            self.alpha4 = 0.5 * self.Y1 * (self.Y1 - 1) * self.c4
            self.alpha5 = self.c5X3 * self.Y3

            self.t += self.tau
            t_series.append(self.t)

        return Y1_molecule, Y2_molecule, Y3_molecule, X2_molecule, t_series

    def plot(self):

        if self.type_index == 1:

            Y1_molecule, Y2_molecule, Y3_molecule, X2_molecule, t_series = self.KMC_direct_method()

            graph_title = 'direct method'

            savefig_title = 'Oreganator3_direct method'
        else:
            Y1_molecule, Y2_molecule, Y3_molecule, X2_molecule, t_series = self.KMC_first_reaction_method()
            graph_title = 'first-reaction method'

            savefig_title = 'Oreganator3_first-reaction method'

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(
            t_series,
            Y1_molecule,
            color='g',
            label='number of Y1 molecules')
        ax1.plot(
            t_series,
            Y2_molecule,
            color='r',
            label='number of Y2 molecules')
        ax1.plot(
            t_series,
            Y3_molecule,
            color='y',
            label='number of Y3 molecules')
        ax1.legend(loc=0)

        ax2 = ax1.twinx()
        ax2.plot(
            t_series,
            X2_molecule,
            color='k',
            label='number of X2 molecules')

        ax1.set_xlabel('time')
        ax1.set_ylim([0, 10000])
        ax2.set_ylim([0, 100000])
        ax1.set_ylabel('number of Y1, Y2 and Y3 molecules')
        ax2.set_ylabel('number of X2 molecules')
        ax2.legend(loc=2)
        plt.title(graph_title)
        plt.savefig(savefig_title)

        plt.show()


# In[26]:


def main_direct_method():

    Oregonator3 = Oregonator_3(6, 100000, 500, 1000, 2000, 2000, 50000, 1)

    Oregonator3.plot()


# In[27]:


main_direct_method()


# In[30]:


def main_first_reaction_method():

    Oregonator3 = Oregonator_3(6, 100000, 500, 1000, 2000, 2000, 50000, 2)

    Oregonator3.plot()


# In[31]:


main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:
