import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

#initialize
N = 100
p = 0.5
k = 4
G = nx.gnp_random_graph(N, 2*k/(N-1))

alpha = 20
beta = 2

for node in G.nodes():
    G.nodes[node]['S'] = np.random.choice([-1, 1])
    G.nodes[node]['C'] = np.random.choice([-1, 1])

#set color for the plot
value_color_map = {1: 'yellow', -1: 'purple'}
colors = [value_color_map[G.nodes[node]['S']] for node in G.nodes()]

nx.draw(G, node_color=colors)
plt.show()

def calculate_neighbor_sum(node):
    neighbor_sum = sum(G.nodes[neighbor]['S'] for neighbor in G.neighbors(node))
    return neighbor_sum

def update_node(node):
    local = calculate_neighbor_sum(node)
    h = local - G.nodes[node]['C'] * alpha * M_t / N
    p = 1/(1 + np.exp(-beta*h))

    #update C
    G.nodes[node]['C'] = -G.nodes[node]['C'] if G.nodes[node]['S'] * G.nodes[node]['C'] * M_t < 0 else G.nodes[node]['C']

    #update S
    rand = random.random()
    if rand < p:
        if G.nodes[node]['S'] == -1:
            G.nodes[node]['S'] == 1
            M_t += 2
    else:
        if G.nodes[node]['S'] == 1:
            G.nodes[node]['S'] == -1
            M_t -= 2

#set variables
M_t = sum(nx.get_node_attributes(G, 'S').values())
print(M_t)

#def update()