#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
from matplotlib import pyplot as plt


# In[77]:


# This on-lattice KMC algorithm simulate the non-reactive surface processes.
class Lattice():

    def __init__(self, t_max, height, width, alpha):

        # KMC simulation time limit.
        self.t_max = t_max

        # Height and width are sizes of the lattice we create.
        self.height = height

        self.width = width

        # The rate constant of events.
        self.alpha = alpha

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

    # This function gives an event list.
    def events_list(self):
        events = []

        for i_site in range(self.N_sites):

            # This site is vacant. The only possible elementary event for it is
            # adsorption.
            if self.lattice_state[i_site] == 0:

                # Number 7 denotes event of adsorption.
                i_site_event = [7]

            # This site is occupied and the events of this site depends on
            # status of its 4 neighbouring sites.
            else:

                # This site is occupied and its four neighbouring sites are vacant.
                # There are 5 possible events for this site. Go up/right/down/left or desorption.
                # Number 6 denotes event of desorption
                # Number 7 denotes event of adsorption.
                # Number8 denotes event of going up.
                # Number9 denotes event of going right.
                # Number10 denotes event of going down.
                # Number11 denotes event of going left.
                if self.neighbours[i_site] == [0, 0, 0, 0]:
                    i_site_event = [8, 9, 10, 11, 6]

                elif self.neighbours[i_site] == [1, 0, 0, 0]:
                    i_site_event = [9, 10, 11, 6]

                elif self.neighbours[i_site] == [0, 1, 0, 0]:
                    i_site_event = [8, 10, 11, 6]

                elif self.neighbours[i_site] == [0, 0, 1, 0]:
                    i_site_event = [8, 9, 11, 6]

                elif self.neighbours[i_site] == [0, 0, 0, 1]:
                    i_site_event = [8, 9, 10, 6]

                elif self.neighbours[i_site] == [1, 1, 0, 0]:
                    i_site_event = [10, 11, 6]

                elif self.neighbours[i_site] == [1, 0, 1, 0]:
                    i_site_event = [9, 11, 6]

                elif self.neighbours[i_site] == [1, 0, 0, 1]:
                    i_site_event = [9, 10, 6]

                elif self.neighbours[i_site] == [0, 1, 1, 0]:
                    i_site_event = [8, 11, 6]

                elif self.neighbours[i_site] == [0, 1, 0, 1]:
                    i_site_event = [8, 10, 6]

                elif self.neighbours[i_site] == [0, 0, 1, 1]:
                    i_site_event = [8, 9, 6]

                elif self.neighbours[i_site] == [1, 1, 1, 0]:
                    i_site_event = [11, 6]

                elif self.neighbours[i_site] == [1, 1, 0, 1]:
                    i_site_event = [10, 6]

                elif self.neighbours[i_site] == [1, 0, 1, 1]:
                    i_site_event = [9, 6]

                elif self.neighbours[i_site] == [0, 1, 1, 1]:
                    i_site_event = [8, 6]

                # This site is occupied and all of its neighbouring sites are
                # occupied.
                else:
                    i_site_event = [6]

            events.append([i_site_event])
        return events

    # This function gives a random waiting time for the selected event.
    def waiting_time_generation(self):

        r1 = np.random.rand()

        waiting_time = - 1.0 / self.sum_alpha * np.log(1.0 - r1)

        return waiting_time

    # This function gives the sum of all rate constants.
    def sum_of_alpha(self):

        length_events = []

        for i_site in range(self.N_sites):

            length_events.append(len(self.events[i_site]))

        # how many events in total
        num_events = np.sum(length_events)

        # Suppose alpha for each event is the same
        sum_alpha = self.alpha * num_events

        return sum_alpha

    # This function determines which event to execute on which site.
    def site_event_selection(self):

        r2 = np.random.rand()

        left = 0

        exit_flag = False

        for i_site in range(self.N_sites):

            for i_event in range(len(self.events[i_site])):

                if left * self.alpha < r2 * \
                        self.sum_alpha < (left + 1) * self.alpha:

                    site_to_change = i_site

                    which_event_to_execute = i_event

                    exit_flag = True

                    break

                else:

                    left += 1

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

        while self.t < self.t_max:

            self.cov = self.coverage()

            cov_all.append(self.cov)

            self.events = self.events_list()

            self.sum_alpha = self.sum_of_alpha()

            self.waiting_time = self.waiting_time_generation()

            self.site_to_change, self.which_event_to_execute = self.site_event_selection()

            self.lattice_state = self.event_executing()

            time_all.append(self.t)

            self.t += self.waiting_time

        average_coverage = np.sum(cov_all) / len(cov_all)

        print('The average covergae is {0}'.format(average_coverage))

        return cov_all, time_all

    def plot(self):
        cov_all, time_all = self.events_progressing()
        plt.figure()
        plt.plot(time_all, cov_all, color='black')
        plt.xlabel('time')
        plt.ylabel('coverage')
        plt.title('direct method (same alpha)')
        plt.savefig('coverage_direct method (same alpha)')
        plt.show()


# In[78]:


def main_direct_method():

    Lattice1 = Lattice(1, 10, 10, 500)

    Lattice1.plot()


# In[79]:

main_direct_method()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
