#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from matplotlib import pyplot as plt


# In[8]:


# This class create a on-lattice KMC model which simulates the
# non-reactive surface processes.
class KMC_lattice():

    def __init__(self, height, width, k_ads_expo, p, k_des):

        # Height and width are sizes of the lattice we create.
        self.height = height

        self.width = width

        # Pre-exponential of adsorption rate constant.
        self.k_ads_expo = k_ads_expo

        # Partial pressure of adsorbates.
        self.p = p

        # Rate constant of desorption.
        self.alpha_ads = self.k_ads_expo * self.p

        # The rate constant of desorption.
        self.alpha_des = k_des

        # Since rate of diffusion has no influence on the results, we assume it
        # is one.
        self.alpha_diff = 1

        # Since for different events, there are different rate constants.
        # We Create a list named "alpha_type" to store different types of
        # alpha.
        self.alpha_type = [self.alpha_ads, self.alpha_des, self.alpha_diff]

        self.t = 0

    # Initialize a vacant lattice based on the given height and width.
    # Create a list named 'lattice_state' which contains the state(vacant or occupied)
    # of all sites on the lattice.
    # If a value in the list 'lattice_state' equals zero, it means the corresponding site
    # is vacant. If it is one, it means the site is occupied.

    def lattice_initialization(self):
        twoD_matrix = np.zeros((self.height, self.width), dtype=np.int)
        lattice_state = []
        for x_pos in range(self.height):
            for y_pos in range(self.width):
                lattice_state.append(twoD_matrix[x_pos, y_pos])
        return twoD_matrix, lattice_state

    # This function gives a list named 'neighbours' which contains the four neighbouring site's
    # states of each site on the lattice.
    # The list 'neighbours' looks like [[0,1,0,1],[0,0,0,0], [1,0,1,0],...]

    def site_neighbours(self):

        neighbours = []
        self.N_sites = np.size(self.twoD_matrix)

        for n in range(self.N_sites):
            # upper-left corner site
            if n == 0:
                up = (self.height - 1) * self.width
                right = n + 1
                down = n + self.width
                left = self.width - 1

            # uppermost boundary sites except lupper-eft and upper-right corner
            # sites
            elif 0 < n < self.width - 1:
                up = n + (self.height - 1) * self.width
                right = n + 1
                down = n + self.width
                left = n - 1

            # upper-right corner site
            elif n == self.width - 1:
                up = self.height * self.width - 1
                right = 0
                down = n + self.width
                left = n - 1

            # bottom-left corner site
            elif n == (self.height - 1) * self.width:
                up = n - self.width
                right = n + 1
                down = 0
                left = self.height * self.width - 1

            # bottom-right corner site
            elif n == self.height * self.width - 1:
                up = n - self.width
                right = (self.height - 1) * self.width
                down = self.width - 1
                left = n - 1

            # bottom sites except bottom-left and bottom-right corner sites
            elif (self.height - 1) * self.width < n < self.height * self.width - 1:
                up = n - self.width
                right = n + 1
                down = n - (self.height - 1) * self.width
                left = n - 1

            # left boundary sites except upper-left and bottom-left corner
            # sites
            elif n % self.width == 0 & n != 0 & n != (self.height - 1) * self.width:
                up = n - self.width
                right = n + 1
                down = n + self.width
                left = n + self.width - 1

            # right boundary sites except upper-right and bottom-right corner
            # sites
            elif (n + 1) % self.width == 0 & n != self.width - 1 & n != self.height * self.width - 1:
                up = n - self.width
                right = n - (self.width - 1)
                down = n + self.width
                left = n - 1

            # inner sites
            else:
                up = n - self.width
                right = n + 1
                down = n + self.width
                left = n - 1

            neighbours.append([up, right, down, left])

        return neighbours, self.N_sites

    def events_list(self):

        events = []

        alpha_list = []

        for i_site in range(self.N_sites):

            # This site is vacant. The only possible elementary event for it is
            # adsorption.
            if self.lattice_state[i_site] == 0:

                # 7 denotes adsorption
                i_site_event = [7]

                alpha_list.append(self.alpha_type[0])

            # This site is occupied and the events of this site depends on
            # status of its 4 neighbouring sites.
            else:

                # This site is occupied and its four neighbouring sites are vacant.
                # There are 5 possible events for this site. Go up/right/down/left or desorption.
                # 0 denotes the vacant site.
                # 1 denotes the occupied site.
                # 6 desorption
                # 7 adsorption
                # 8 go up
                # 9 go right
                # 10 go down
                # 11 left
                if self.neighbours[i_site] == [0, 0, 0, 0]:
                    i_site_event = [8, 9, 10, 11, 6]
                    # with no occupied neighbouring site
                    alpha_list.append(self.alpha_type[2])
                    alpha_list.append(self.alpha_type[2])
                    alpha_list.append(self.alpha_type[2])
                    alpha_list.append(self.alpha_type[2])
                    alpha_list.append(self.alpha_type[1])

                else:
                    if np.sum(self.neighbours[i_site]) == 1:

                        if self.neighbours[i_site] == [1, 0, 0, 0]:
                            i_site_event = [9, 10, 11, 6]

                        elif self.neighbours[i_site] == [0, 1, 0, 0]:
                            i_site_event = [8, 10, 11, 6]

                        elif self.neighbours[i_site] == [0, 0, 1, 0]:
                            i_site_event = [8, 9, 11, 6]

                        else:
                            # status_i_neighbours == [0,0,0,1]:
                            i_site_event = [8, 9, 10, 6]

                        # with one occupied neighbouring site
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[1])

                    elif np.sum(self.neighbours[i_site]) == 2:

                        if self.neighbours[i_site] == [1, 1, 0, 0]:
                            i_site_event = [10, 11, 6]

                        elif self.neighbours[i_site] == [1, 0, 1, 0]:
                            i_site_event = [9, 11, 6]

                        elif self.neighbours[i_site] == [1, 0, 0, 1]:
                            i_site_event = [9, 10, 6]

                        elif self.neighbours[i_site] == [0, 1, 1, 0]:
                            i_site_event = [8, 11, 6]

                        elif self.neighbours[i_site] == [0, 1, 0, 1]:
                            i_site_event = [8, 10, 6]

                        else:
                            # status_i_neighbours == [0,0,1,1]:
                            i_site_event = [8, 9, 6]

                        # with two occupied neighbouring site
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[1])

                    elif np.sum(self.neighbours[i_site]) == 3:

                        if self.neighbours[i_site] == [1, 1, 1, 0]:
                            i_site_event = [11, 6]

                        elif self.neighbours[i_site] == [1, 1, 0, 1]:
                            i_site_event = [10, 6]

                        elif self.neighbours[i_site] == [1, 0, 1, 1]:
                            i_site_event = [9, 6]

                        else:
                            # status_i_neighbours == [0,1,1,1]:
                            i_site_event = [8, 6]

                        # with three occupied neighbouring site
                        alpha_list.append(self.alpha_type[2])
                        alpha_list.append(self.alpha_type[1])

                    # This site is occupied and all of its neighbouring sites are occupied.
                    # The only possible elementary event is desorption.
                    else:
                        i_site_event = [6]
                        alpha_list.append(self.alpha_type[1])

            events.append(i_site_event)

        return events, alpha_list

    # This function gives sum of rate constants of all events.
    def sum_of_alpha(self):

        sum_alpha = np.sum(self.alpha_list)

        return sum_alpha

    # This function gives a random waiting time for the selected event.
    def waiting_time_generation(self):

        r1 = np.random.rand()

        waiting_time = - 1.0 / self.sum_alpha * np.log(1.0 - r1)

        return waiting_time

    # Direct method.

    def site_event_selection(self):

        r2 = np.random.rand()

        left_side = 0

        medium = 0

        right_side = self.alpha_list[medium]

        exit_flag = False

        for i_site in range(self.N_sites):

            for i_event in range(len(self.events[i_site])):

                if left_side < r2 * self.sum_alpha < right_side:

                    site_to_change = i_site

                    which_event_to_execute = i_event

                    exit_flag = True

                    break

                else:

                    left_side = right_side

                    medium += 1

                    right_side += self.alpha_list[medium]

            if exit_flag:

                break

        return site_to_change, which_event_to_execute

    # Once an event is executed, this function updates the list
    # 'lattice_state'.
    def event_executing(self):

        event_chosen = self.events[self.site_to_change][self.which_event_to_execute]

        # desorption
        if event_chosen == 6:
            self.lattice_state[self.site_to_change] = 0

        # adsorption
        elif event_chosen == 7:
            self.lattice_state[self.site_to_change] = 1

        # go up
        elif event_chosen == 8:
            self.lattice_state[site_to_change] = 0
            site_influenced = self.neighbours[self.site_to_change][0]
            self.lattice_state[site_influenced] = 1

        # go right
        elif event_chosen == 9:
            self.lattice_state[self.site_to_change] = 0
            site_influenced = self.neighbours[self.site_to_change][1]
            self.lattice_state[site_influenced] = 1

        # go down
        elif event_chosen == 10:
            self.lattice_state[site_to_change] = 0
            site_influenced = self.neighbours[self.site_to_change][2]
            self.lattice_state[site_influenced] = 1

        # go left (event_chosen == 11)
        else:
            self.lattice_state[self.site_to_change] = 0
            site_influenced = self.neighbours[self.site_to_change][3]
            self.lattice_state[site_influenced] = 1

        return self.lattice_state

    # This function calculates the coverage of the lattice.
    def coverage(self):

        num_occupied = np.sum(self.lattice_state)

        cov = num_occupied / len(self.lattice_state)

        return cov

    def events_progressing(self):

        self.twoD_matrix, self.lattice_state = self.lattice_initialization()

        self.neighbours, self.N_sites = self.site_neighbours()

        cov_all = []

        time_all = []

        for i in range(10000):

            self.cov = self.coverage()

            cov_all.append(self.cov)

            self.events, self.alpha_list = self.events_list()

            self.sum_alpha = self.sum_of_alpha()

            self.waiting_time = self.waiting_time_generation()

            self.site_to_change, self.which_event_to_execute = self.site_event_selection()

            self.lattice_state = self.event_executing()

            time_all.append(self.t)

            self.t += self.waiting_time

        average_coverage = np.sum(cov_all) / len(cov_all)

        print('The average covergae from KMC simulation is {0}.'.format(average_coverage))

        return cov_all, time_all

    def plot(self):
        cov_all, time_all = self.events_progressing()
        plt.figure()
        plt.plot(time_all, cov_all, color='black')
        plt.xlabel('time')
        plt.ylabel('coverage')
        plt.title('Simulation results of KMC')
        plt.savefig('average coverage from KMC simulation')
        plt.show()


# In[ ]:


# In[ ]:
