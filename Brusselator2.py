#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from matplotlib import pyplot as plt


# In[32]:


class Brusselator_2():

    def __init__(self, t_max, X1, c1, c2X2, c3, c4, Y1, Y2):

        # Now t_max is not the KMC simulation time limit but the loop times
        # limit.
        self.t_max = t_max

        self.X1 = X1

        self.c1 = c1

        self.c2X2 = c2X2

        self.c3 = c3

        self.c4 = c4

        self.Y1 = Y1

        self.Y2 = Y2

        # alpha1 = h1 * c1 = X1 * c1
        self.alpha1 = self.X1 * self.c1

        # alpha2 = h2 * c2 = X2 * Y1 * c2 = c2X2 * Y1
        self.alpha2 = self.c2X2 * self.Y1

        # alpha3 = h3 * c3 = Y2 * Y1 * (Y1 -1) * 0.5 * c3
        self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3

        # alpha4 = h4 * c4 = Y1 * c4
        self.alpha4 = self.Y1 * self.c4

        self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4

        self.t = 0

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
        X1_molecule = [self.X1]
        t_series = [self.t]

        for i in range(self.t_max):

            if self.alpha0 == 0:

                break

            self.tau = self.interval_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:
                self.X1 -= 1
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
            X1_molecule.append(self.X1)

            self.alpha1 = self.X1 * self.c1
            self.alpha2 = self.c2X2 * self.Y1
            self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3
            self.alpha4 = self.Y1 * self.c4

            self.alpha0 = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4

            self.t += self.tau
            t_series.append(self.t)

        return Y1_molecule, Y2_molecule, X1_molecule, t_series

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
        X1_molecule = [self.X1]
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
                self.X1 -= 1
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
            X1_molecule.append(self.X1)

            self.alpha1 = self.X1 * self.c1
            self.alpha2 = self.c2X2 * self.Y1
            self.alpha3 = self.Y2 * self.Y1 * (self.Y1 - 1) * 0.5 * self.c3
            self.alpha4 = self.Y1 * self.c4

            self.t += self.tau
            t_series.append(self.t)

        return Y1_molecule, Y2_molecule, X1_molecule, t_series


# In[33]:


def plot_direct_method1(Y1_molecule, Y2_molecule, X1_molecule, t_series):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(t_series, Y1_molecule, color='g', label='number of Y1 molecules')

    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.plot(t_series, X1_molecule, color='k', label='number of X1 molecules')

    ax1.set_xlabel('time')
    ax1.set_ylim([0, 18000])
    ax2.set_ylim([0, 100000])
    ax1.set_ylabel('number of Y1 molecules')
    ax2.set_ylabel('number of X1 molecules')
    plt.xlim((0, 35))
    ax2.legend(loc=4)
    ax1.text(30, 14000, 'a')

    plt.title('direct method:X1 vs. Y1')
    plt.savefig('Brusselator2_direct method1')

    plt.show()


# In[34]:


def plot_direct_method2(Y1_molecule, Y2_molecule, X1_molecule, t_series):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(t_series, Y2_molecule, color='g', label='number of Y2 molecules')

    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.plot(t_series, X1_molecule, color='k', label='number of X1 molecules')

    ax1.set_xlabel('time')
    ax1.set_ylim([0, 18000])
    ax2.set_ylim([0, 100000])
    ax1.set_ylabel('number of Y2 molecules')
    ax2.set_ylabel('number of X1 molecules')
    plt.xlim((0, 35))
    ax2.legend(loc=4)
    ax1.text(30, 14000, 'b')

    plt.title('direct method:X1 vs. Y2')
    plt.savefig('Brusselator2_direct method2')

    plt.show()


# In[35]:


def main_direct_method():

    Brusselator2 = Brusselator_2(
        2000000, 100000, 0.05, 50, 0.00005, 5, 1000, 2000)

    Y1_molecule, Y2_molecule, X1_molecule, t_series = Brusselator2.KMC_direct_method()

    plot_direct_method1(Y1_molecule, Y2_molecule, X1_molecule, t_series)

    plot_direct_method2(Y1_molecule, Y2_molecule, X1_molecule, t_series)


# In[36]:


main_direct_method()


# In[37]:


def plot_first_reaction_method1(
        Y1_molecule,
        Y2_molecule,
        X1_molecule,
        t_series):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(t_series, Y1_molecule, color='g', label='number of Y1 molecules')

    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.plot(t_series, X1_molecule, color='k', label='number of X1 molecules')

    ax1.set_xlabel('time')
    ax1.set_ylim([0, 18000])
    ax2.set_ylim([0, 100000])
    ax1.set_ylabel('number of Y1 molecules')
    ax2.set_ylabel('number of X1 molecules')
    plt.xlim((0, 35))
    ax2.legend(loc=4)
    ax1.text(30, 14000, 'a')

    plt.title('first-reaction method: X1 vs. y1')
    plt.savefig('Brusselator2_first-reaction method1')

    plt.show()


# In[38]:


def plot_first_reaction_method2(
        Y1_molecule,
        Y2_molecule,
        X1_molecule,
        t_series):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(t_series, Y2_molecule, color='g', label='number of Y2 molecules')

    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.plot(t_series, X1_molecule, color='k', label='number of X1 molecules')

    ax1.set_xlabel('time')
    ax1.set_ylim([0, 18000])
    ax2.set_ylim([0, 100000])
    ax1.set_ylabel('number of Y2 molecules')
    ax2.set_ylabel('number of X1 molecules')
    plt.xlim((0, 35))
    ax2.legend(loc=4)
    ax1.text(30, 14000, 'b')

    plt.title('first-reaction: X1 vs. Y2')
    plt.savefig('Brusselator2_first-reaction method2')

    plt.show()


# In[39]:


def main_first_reaction_method():

    Brusselator2 = Brusselator_2(
        2000000, 100000, 0.05, 50, 0.00005, 5, 1000, 2000)

    Y1_molecule, Y2_molecule, X1_molecule, t_series = Brusselator2.KMC_first_reaction_method()

    plot_first_reaction_method1(
        Y1_molecule,
        Y2_molecule,
        X1_molecule,
        t_series)

    plot_first_reaction_method2(
        Y1_molecule,
        Y2_molecule,
        X1_molecule,
        t_series)


# In[40]:


main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
