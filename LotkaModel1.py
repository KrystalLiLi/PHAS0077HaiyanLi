#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from matplotlib import pyplot as plt


# In[19]:


# In this class, number of X is fixed.
# Parameter "type_index" decides if the method is direct method or first-reaction method.
# If type_index equals one, it means direct method is used. If it is two,
# first-reaction is used.
class LotkaModel_1():

    def __init__(self, t_max, c1X, c2, c3, Y1, Y2, type_index):

        self.t_max = t_max

        self.c1X = c1X

        self.c2 = c2

        self.c3 = c3

        self.Y1 = Y1

        self.Y2 = Y2

        # alpha1 = h1 * c1 = X * Y1 * c1 = c1 * X * Y1 = constant * Y1
        self.alpha1 = self.c1X * self.Y1

        # h2 = Y1 * Y2
        self.alpha2 = self.Y1 * self.Y2 * self.c2

        # h3 = Y2
        self.alpha3 = self.Y2 * self.c3

        self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3

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

        elif self.alpha1 + self.alpha2 < r2 * self.alpha0 < self.alpha0:

            mu = 3

        return mu

    def KMC_direct_method(self):

        Y1_molecule = [self.Y1]
        Y2_molecule = [self.Y2]
        t_series = [self.t]

        while self.t < self.t_max:

            self.tau = self.interval_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            else:
                self.Y2 -= 1
            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha1 = self.c1X * self.Y1
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.Y2 * self.c3

            self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3

            if self.alpha0 == 0:

                self.t += self.tau
                t_series.append(self.t)

                break

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

        while self.t < self.t_max:

            self.alpha_list = [self.alpha1, self.alpha2, self.alpha3]

            self.tau, self.mu = self.first_reaction_method()

            if self.mu == 1:
                self.Y1 += 1
            elif self.mu == 2:
                self.Y1 -= 1
                self.Y2 += 1
            else:
                self.Y2 -= 1

            Y1_molecule.append(self.Y1)
            Y2_molecule.append(self.Y2)

            self.alpha1 = self.c1X * self.Y1
            self.alpha2 = self.Y1 * self.Y2 * self.c2
            self.alpha3 = self.Y2 * self.c3

            self.t += self.tau
            t_series.append(self.t)

            if self.alpha1 == 0 or self.alpha2 == 0 or self.alpha3 == 0:
                break

        return Y1_molecule, Y2_molecule, t_series

    def plot(self):

        if self.type_index == 1:

            Y1_molecule, Y2_molecule, t_series = self.KMC_direct_method()

            graph_title1 = 'direct method'
            graph_title2 = 'direct method'
            savefig_title = 'LotkaModel1_direct method'
        else:
            Y1_molecule, Y2_molecule, t_series = self.KMC_first_reaction_method()
            graph_title1 = 'first-reaction method'
            graph_title2 = 'first-reaction method'
            savefig_title = 'LotkaModel1_first-reaction method'

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.plot(t_series[0:300000], Y1_molecule[0:300000],
                 color='green', label='number of Y1 molecules')
        plt.plot(t_series[0:300000], Y2_molecule[0:300000],
                 color='yellow', label='number of Y2 molecules')
        plt.hlines(1000, 0, 10, color="k", linestyle='dashed')
        plt.xlabel('time')
        plt.ylabel('numbers of Y1 and Y2 molecules')
        plt.ylim((0, 3000))
        plt.legend()
        plt.text(0, 200, 'a')
        plt.title(graph_title1)

        plt.subplot(2, 2, 2)
        plt.plot(
            t_series,
            Y1_molecule,
            color='green',
            label='number of Y1 molecules')
        plt.hlines(1000, 0, 30, color="k", linestyle='dashed')
        plt.xlabel('time')
        plt.ylabel('number of Y1 molecules')
        plt.legend()
        plt.ylim((0, 6000))
        plt.text(2, 4000, 'b')
        plt.title(graph_title2)

        plt.subplots_adjust(wspace=0.5, hspace=0)
        plt.savefig(savefig_title)
        plt.show()


# In[20]:


def main_direct_method():

    LotkaModel1 = LotkaModel_1(30, 10, 0.01, 10, 1000, 1000, 1)

    LotkaModel1.plot()


# In[21]:


main_direct_method()


# In[22]:


def main_first_reaction_method():

    LotkaModel1 = LotkaModel_1(30, 10, 0.01, 10, 1000, 1000, 2)

    LotkaModel1.plot()


# In[23]:


main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:
