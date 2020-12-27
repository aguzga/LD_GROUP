#!/usr/bin/env python
#cython: language_level=3, boundscheck=False, wraparound = False


# In[1]:


import numpy as np
import math as m
import networkx as nx
from time import time
cimport numpy

# In[2]:


cpdef float get_stimulus(float A, float beta, float reward):
    cdef float arg
    arg = beta * (reward - A)
    return m.tanh(arg)

# In[3]:

cpdef float update_p(float aspiration_level, float beta, float reward, int prev_action, float p):
    cdef float s = get_stimulus(aspiration_level, beta, reward)
    if prev_action == 0:  #previous action = cooperation
        if s >= 0:
            return p + (1 - p) * s
        else:
            return p + p * s
    else:  #previous action = defect
        if s >= 0:
            return p - p * s
        else:
            return p - (1 - p) * s

# In[4]:


cpdef float misimplement_prob_update(float eps, float prob):
    return prob * (1 - eps) + (1 - prob) * eps

# In[5]:


def get_payoff_matrix(b, c):
    return np.array([[b - c, -c],
                     [b, 0]])

# In[6]:


def get_payoffs(p1, p2, payoff_mat):
    return payoff_mat[p1][p2]

# In[7]:


def create_ring_graph(N, k):
    return nx.watts_strogatz_graph(N, k, 0)

#pos = nx.circular_layout(G_small_ring)
#plt.figure(3,figsize=(20,20))
#nx.draw_networkx(G_small_ring, pos=pos, with_labels=False)


# In[8]:


cpdef float simulate_game(G, int k, int rounds, float A, float beta, float eps, int b, int c):
    cdef int nodes = len(list(G.nodes))
    cdef numpy.ndarray[numpy.int_t, ndim=1] actions
    cdef numpy.ndarray[numpy.float_t, ndim=1] probas
    cdef numpy.ndarray[numpy.float_t, ndim=1] counts
    cdef numpy.ndarray[numpy.float_t, ndim=1] payoffs
    cdef numpy.ndarray[numpy.int_t, ndim=2] neighbours
    cdef numpy.ndarray[numpy.int_t, ndim=2] payoff_mat

    cdef numpy.ndarray[numpy.int_t, ndim=1] n_def
    cdef numpy.ndarray[numpy.int_t, ndim=1] n_coop
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] neg
    cdef numpy.ndarray[numpy.float_t, ndim=1] stim

    cdef numpy.ndarray[numpy.uint8_t, ndim=1] m_def
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] m_coop
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] m_stimp
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] m_stimn

    cdef numpy.ndarray[numpy.uint8_t, ndim=1] idx1
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] idx2
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] idx3
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] idx4

    probas = np.array([0.8 for _ in range(nodes)])
    counts = np.zeros(rounds)
    assortment = np.zeros(rounds)

    payoff_mat = np.array([[b - c, -c],
                           [b, 0]])
    cdef float new_p
    cdef int _k = k
    cdef Py_ssize_t n
    nodes_list = list(G.nodes)
    neighbours = np.array([list(G.neighbors(node)) for node in range(nodes)])
    cdef Py_ssize_t r = 0
    cdef Py_ssize_t _rounds = 50
    cdef Py_ssize_t node = 0
    cdef Py_ssize_t neighbour = 0
    cdef int countC
    cdef int countD
    for r in range(_rounds):
        payoffs = np.zeros(nodes)
        actions = np.random.binomial(1, p=(1 - probas))
        countC = 0
        countD = 0

        n_def = np.count_nonzero(actions[neighbours], axis=1)
        n_coop = k - n_def

        neg = np.logical_not(actions)
        countC = (neg * n_coop).sum()
        countD = (actions * n_coop).sum()
        payoffs = (n_coop * payoff_mat[actions][:, 0] + n_def * payoff_mat[actions][:, 1]).astype(np.float)
        payoffs = payoffs / k

        stim = np.tanh(beta * (payoffs - A))

        m_def = (actions != 0)
        m_coop = ~m_def

        m_stimp = (stim >= 0)
        m_stimn = ~m_stimp

        idx1 = m_coop & m_stimp
        probas[idx1] += (1 - probas[idx1]) * stim[idx1]

        idx2 = m_coop & m_stimn
        probas[idx2] += probas[idx2] * stim[idx2]

        idx3 = m_def & m_stimp
        probas[idx3] -= probas[idx3] * stim[idx3]

        idx4 = m_def & m_stimn
        probas[idx4] -= (1 - probas[idx4]) * stim[idx4]

        probas = probas * (1 - eps) + (1 - probas) * eps

        counts[r] = nodes - np.count_nonzero(actions)
        assortment[r] = (countC - countD) / (nodes * k)
    assort = np.sum(assortment) / rounds
    counts = counts / nodes
    return counts[rounds - 1]


def run_test():
    cdef int N = 100
    cdef int k = 6
    G = create_ring_graph(N, k)
    cdef float A = 1.0
    cdef float beta = 0.2
    cdef float eps = 0.05
    cdef int b = 6
    cdef int c = 1

    startTime = time()
    cdef numpy.ndarray[numpy.float_t, ndim=1] A_values
    cdef numpy.ndarray[numpy.float_t, ndim=1] eps_values
    A_values = np.linspace(-1, 5, num=100)
    eps_values = np.linspace(0, 0.5, num=100)
    cdef numpy.ndarray[numpy.float_t, ndim=2] heatmap
    heatmap = np.zeros((100, 100))

    cdef Py_ssize_t max_range = 100

    cdef Py_ssize_t a_i = 0
    cdef Py_ssize_t eps_i = 0
    for a_i in range(max_range):
        for eps_i in range(max_range):
            heatmap[eps_i][a_i] = simulate_game(G, k, 50, A_values[a_i], beta, eps_values[eps_i], b, c)

    endTime = time()
    print("\nSimulating took {} seconds".format(round(endTime - startTime)))

#n_def = np.count_nonzero(actions[neighbours[node]])
#n_coop = _k-n_def
#neg = np.logical_not(actions[node])
#countC += np.logical_not(actions[node]) * n_coop
#countD += np.logical_not(actions[node]) * n_def
#payoffs[node] += n_coop * payoff_mat[actions[node]][0] + n_def * payoff_mat[actions[node]][1]
