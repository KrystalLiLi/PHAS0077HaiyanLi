#!/usr/bin/env python
# coding: utf-8

# $\bar{X} + Y \stackrel{c_1}{\longrightarrow}2Y$
# $2Y \stackrel{c_2}{\longrightarrow}Z$

# In[17]:


import numpy as np
from matplotlib import pyplot as plt


# In[18]:


class SimpleSystem:

    # Initialization
    def __init__(self, Y, c2, c1X, t_max):

        self.Y = Y

        self.c2 = c2

        self.c1X = c1X

        self.t_max = t_max

        self.alpha1 = self.c1X * self.Y

        self.h2 = self.Y * (self.Y - 1) * 0.5

        self.alpha2 = self.c2 * self.h2

        self.alpha0 = self.alpha1 + self.alpha2

        self.t = 0

    # This function gives the interval time for the event to be executed.
    def waiting_time_direct_method(self):

        r1 = np.random.rand()

        tau = - 1.0 / self.alpha0 * np.log(1.0 - r1)

        return tau

    # This function uses direct method to give the index of the event to be
    # executed.
    def event_selection_direct_method(self):

        mu = 0

        r2 = np.random.rand()

        if 0 < r2 * self.alpha0 < self.alpha1:

            mu = 1

        elif self.alpha1 < r2 * self.alpha0 < self.alpha1:

            mu = 2

        return mu

    # This function enters the KMC loop.
    def KMC_direct_method(self):

        Y_molecule = [self.Y]

        t_series = [self.t]

        while self.t < self.t_max:

            self.tau = self.waiting_time_direct_method()

            self.mu = self.event_selection_direct_method()

            if self.mu == 1:

                self.Y += 1

            else:

                self.Y -= 2

            Y_molecule.append(self.Y)

            self.alpha1 = self.c1X * self.Y

            self.alpha2 = self.c2 * self.Y * (self.Y - 1) * 0.5

            self.alpha0 = self.alpha1 + self.alpha2

            self.t += self.tau

            t_series.append(self.t)

        return Y_molecule, t_series

    # This function uses first-reaction method to determine which event to excute
    # and the corresponding waiting time.
    def first_reaction_method(self):

        random_time = []

        alpha_list = [self.alpha1, self.alpha2]

        for i in range(2):

            r1 = np.random.rand()

            interval_time = - 1.0 / alpha_list[i] * np.log(1.0 - r1)

            random_time.append(interval_time)

        tau = min(random_time)

        mu = random_time.index(tau) + 1

        return tau, mu

    def KMC_first_reaction_method(self):

        Y_molecule = [self.Y]

        t_series = [self.t]

        while self.t < self.t_max:

            self.tau, self.mu = self.first_reaction_method()

            if self.mu == 1:

                self.Y += 1

            else:

                self.Y -= 2

            Y_molecule.append(self.Y)

            self.alpha1 = self.c1X * self.Y

            self.alpha2 = self.c2 * self.Y * (self.Y - 1) * 0.5

            self.t += self.tau

            t_series.append(self.t)

        return Y_molecule, t_series


# In[19]:


def plot_direct_method(t_series1, t_series2, Y_molecule1, Y_molecule2):

    plt.figure()

    plt.plot(t_series1, Y_molecule1, color='green', label='Y1 = 10')

    plt.plot(t_series2, Y_molecule2, color='red', label='Y2 = 3000')

    plt.legend()

    plt.hlines(1000, 0, 5, color="k", linestyle='dashed')

    plt.xlabel('time')

    plt.ylabel('number of Y molecules')

    plt.title('direct method')

    plt.savefig('SimpleSystem-direct method')

    plt.show()


# In[20]:


def main_direct_method():

    simple_system_Y1 = SimpleSystem(10, 0.005, 5, 5)

    simple_system_Y2 = SimpleSystem(3000, 0.005, 5, 5)

    Y_molecule1, t_series1 = simple_system_Y1.KMC_direct_method()

    Y_molecule2, t_series2 = simple_system_Y2.KMC_direct_method()

    plot_direct_method(t_series1, t_series2, Y_molecule1, Y_molecule2)


# In[21]:


main_direct_method()


# In[22]:


def plot_first_reaction_method(t_series3, t_series4, Y_molecule3, Y_molecule4):

    plt.figure()

    plt.plot(t_series3, Y_molecule3, color='green', label='Y1 = 10')

    plt.plot(t_series4, Y_molecule4, color='red', label='Y2 = 3000')

    plt.legend()

    plt.hlines(1000, 0, 5, color="k", linestyle='dashed')

    plt.xlabel('time')

    plt.ylabel('number of Y molecules')

    plt.title('first-reaction method')

    plt.savefig('SimpleSystem-first-reaction method')

    plt.show()


# In[23]:


def main_first_reaction_method():

    simple_system_Y1 = SimpleSystem(10, 0.005, 5, 5)

    simple_system_Y2 = SimpleSystem(3000, 0.005, 5, 5)

    Y_molecule3, t_series3 = simple_system_Y1.KMC_first_reaction_method()

    Y_molecule4, t_series4 = simple_system_Y2.KMC_first_reaction_method()

    plot_first_reaction_method(t_series3, t_series4, Y_molecule3, Y_molecule4)


# In[24]:


main_first_reaction_method()


# In[ ]:


# In[ ]:


# In[ ]:
