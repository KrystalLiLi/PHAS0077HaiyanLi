#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
from matplotlib import pyplot as plt


# In[38]:


class LotkaMode2():

    def __init__(self, t_max, c1, c2, c3, X, Y1, Y2, type_index):

        self.t_max = t_max

        self.c1 = c1

        self.c2 = c2

        self.c3 = c3

        self.X = X

        self.Y1 = Y1

        self.Y2 = Y2

        self.h1 = self.X * self.Y1

        self.h2 = self.Y1 * self.Y2

        self.h3 = self.Y2

        self.alpha1 = self.c1 * self.h1

        self.alpha2 = self.c2 * self.h2

        self.alpha3 = self.c3 * self.h3

        self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3

        self.t = 0

        self.type_index = type_index

    def waiting_time_direct_method(self):

        r1 = np.random.rand()

        tau = - 1.0 / self.alpha0 * np.log(1.0 - r1)

        return tau

    def event_selection_direct_method(self):

        r2 = np.random.rand()

        if 0 < r2 * self.alpha0 < self.alpha1:
            mu = 1
        elif self.alpha1 < r2 * self.alpha0 < self.alpha1 + self.alpha2:
            mu = 2
        elif self.alpha1 + self.alpha2 < r2 * self.alpha0 < self.alpha0:
            mu = 3
        return mu

    def KMC_direct_method(self):

        X_molecule = [self.X]

        Y1_molecule = [self.Y1]

        Y2_molecule = [self.Y2]

        t_series = [self.t]

        while self.t < self.t_max:

            if self.alpha0 == 0:

                break

            self.tau = self.waiting_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:
                self.X -= 1
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            else:
                self.Y2 -= 1

            X_molecule.append(self.X)
            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha1 = self.c1 * self.X * self.Y1
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.Y2 * self.c3

            self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3

            self.t += self.tau

            t_series.append(self.t)

        return X_molecule, Y1_molecule, Y2_molecule, t_series

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

        X_molecule = [self.X]

        Y1_molecule = [self.Y1]

        Y2_molecule = [self.Y2]

        t_series = [self.t]

        while self.t < self.t_max:

            self.alpha_list = [self.alpha1, self.alpha2, self.alpha3]

            if self.alpha1 == 0 or self.alpha2 == 0 or self.alpha3 == 0:

                break

            self.tau, self.mu = self.first_reaction_method()

            if self.mu == 1:
                self.X -= 1
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            else:
                self.Y2 -= 1

            X_molecule.append(self.X)
            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha1 = self.c1 * self.X * self.Y1
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.Y2 * self.c3

            self.t += self.tau

            t_series.append(self.t)

        return X_molecule, Y1_molecule, Y2_molecule, t_series

    def plot(self):

        if self.type_index == 1:

            X_molecule, Y1_molecule, Y2_molecule, t_series = self.KMC_direct_method()

            graph_title = 'direct method'
            savefig_title = 'LotkaModel2_direct method_'
        else:
            X_molecule, Y1_molecule, Y2_molecule, t_series = self.KMC_first_reaction_method()
            graph_title = 'first-reaction method'
            savefig_title = 'LotkaModel2_first-reaction method_'

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
            color='y',
            label='number of Y2 molecules')
        ax1.set_ylim([0, 3000])
        ax1.legend(loc=0)
        ax2 = ax1.twinx()
        ax2.plot(
            t_series,
            X_molecule,
            color='k',
            label='number of X molecules')

        ax1.set_xlabel('time')
        ax1.set_ylabel('number of Y1 and Y2 molecules')
        ax2.set_ylabel('number of X molecules')
        ax2.legend(loc=3)

        plt.title(graph_title)
        plt.savefig(savefig_title)
        plt.show()


# In[39]:


def main_direct_method():
    LotkaModel2 = LotkaMode2(30, 0.0001, 0.01, 10, 100000, 1000, 1000, 1)
    LotkaModel2.plot()


# In[40]:


main_direct_method()


# In[41]:


def main_first_reaction_method():
    LotkaModel2 = LotkaMode2(30, 0.0001, 0.01, 10, 100000, 1000, 1000, 2)
    LotkaModel2.plot()


# In[42]:


main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:
