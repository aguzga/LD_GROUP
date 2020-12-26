#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m
import networkx as nx
from time import time
cimport numpy


# In[2]:


cpdef float get_stimulus(float A, float beta, float reward):
    cdef float arg
    arg = beta*(reward - A)
    return m.tanh(arg)


# In[3]:

cdef float update_p(float aspiration_level, float beta, float reward, int prev_action, float p):
    cdef float s = get_stimulus(aspiration_level, beta, reward)
    if prev_action == 0: #previous action = cooperation
        if s >= 0:
            return p + (1-p)*s
        else:
            return p + p*s
    else: #previous action = defect
        if s >= 0:
            return p - p*s
        else:
            return p - (1-p)*s


# In[4]:


cdef float misimplement_prob_update(float eps, float prob):
    return prob*(1-eps) + (1-prob)*eps


# In[5]:


def get_payoff_matrix(b,c):
    return np.array([[b-c, -c],
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
    probas = np.array([0.8 for _ in range(nodes)])
    counts = np.zeros(rounds)
    cdef numpy.ndarray[numpy.float_t, ndim=1] assortment
    assortment = np.zeros(rounds)

    payoff_mat = np.array([[b - c, -c],
              [b, 0]])
    cdef float new_p
    cdef int _k = k
    cdef Py_ssize_t n
    cdef Py_ssize_t node
    nodes_list = list(G.nodes)
    neighbours = np.array([list(G.neighbors(node)) for node in range(nodes)])
    cdef Py_ssize_t r = 0
    cdef Py_ssize_t _rounds = 25

    cdef Py_ssize_t neighbour = 0
    cdef int countC
    cdef int countD
    for r in range(_rounds):
        payoffs = np.zeros(nodes)
        actions = np.random.binomial(1, p = (1 - probas))
        countC = 0
        countD = 0
        for node in range(nodes):
            neighbour = 0
            for neighbour in range(_k):#neighbours[node]:
                #print(neighbours[node])
                n = neighbours[node][neighbour]
                if(actions[n]==0 and actions[node]==0):
                    countC+=1
                elif(actions[n]==0 and actions[node]==1):
                    countD+=1
                payoffs[node] += payoff_mat[actions[node]][actions[n]]#get_payoffs(actions[node], actions[n], payoff_mat)
            payoffs[node] = payoffs[node]/_k
            new_p = update_p(A, beta, payoffs[node], actions[node], probas[node])
            probas[node] = misimplement_prob_update(eps, new_p)
        #node = 0
        #for node in range(nodes):
            #cdef float new_p = update_p(A, beta, payoffs[node], actions[node], probas[node])
            #probas[node] = misimplement_prob_update(eps, new_p)
        counts[r] = nodes - np.count_nonzero(actions)
        assortment[r] = (countC-countD)/(nodes*_k)
    assort = np.sum(assortment)/_rounds
    #print(assort)
    counts = counts / nodes
    return counts[_rounds-1]


def run_test():
    cdef int N = 100
    cdef int k = 2
    G = create_ring_graph(N, k)
    cdef float A = 1.0
    cdef float beta = 0.2
    cdef float eps = 0.05
    cdef int b = 6
    cdef int c = 1

    startTime = time()
    cdef numpy.ndarray[numpy.float_t, ndim=1] A_values
    cdef numpy.ndarray[numpy.float_t, ndim=1] eps_values
    A_values = np.linspace(-1, 5, num = 100)
    eps_values = np.linspace(0, 0.5, num = 100)
    cdef numpy.ndarray[numpy.float_t, ndim=2] heatmap
    heatmap = np.zeros((100,100))

    cdef Py_ssize_t max_range = 100

    cdef Py_ssize_t a_i = 0
    cdef Py_ssize_t eps_i = 0
    for a_i in range(max_range):
        for eps_i in range(max_range):
            heatmap[eps_i][a_i] = simulate_game(G, k, 50, A_values[a_i], beta, eps_values[eps_i], b, c)

    endTime = time()
    print("\nSimulating took {} seconds".format(round(endTime - startTime)))

