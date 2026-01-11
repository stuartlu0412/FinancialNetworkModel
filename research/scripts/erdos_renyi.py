import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

def run_simulation(alpha = 10, beta = 0.7):

    #initialize
    seed = 123
    random.seed(seed)
    N = 50000
    p = 0.5
    k = 4 #average degree is 4
    G = nx.gnp_random_graph(N, 2*k/(N-1), seed = seed)

    #variables that records all the values for further analysis
    M_t = 0 # magnetization
    F_t = 0 # number of fundamentalists
    M_t_values = []
    F_t_values = []

    #initialize the market
    for node in G.nodes():
            G.nodes[node]['S'] = np.random.choice([-1, 1])
            G.nodes[node]['C'] = np.random.choice([-1, 1])

    M_t = sum(nx.get_node_attributes(G, 'S').values())
    F_t = sum([G.nodes[node]['C'] for node in G.nodes if G.nodes[node]['C'] == 1])

    M_t_values.append(M_t)
    F_t_values.append(F_t)

    

