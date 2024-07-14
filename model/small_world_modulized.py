import numpy as np
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from bornholdt_modulized import BornholdtModel

SAVE_PATH = f'/Users/stuartlu/Documents/Complex System/Financial Network Model/results/Small_World_Model_Results'


class SmallWorldModel(BornholdtModel):

    def __init__(self, alpha=20, beta=0.8):
        #initialize
        self.seed = 123
        random.seed(self.seed)
        self.N = 50000
        self.p = 0.5
        self.k = 4 #average degree is 4
        self.G = nx.watts_strogatz_graph(self.N, self.k, 0.5, seed = self.seed)
        
        #self.pos = nx.nx_pydot.pydot_layout(self.G)
        self.alpha = alpha
        self.beta = beta
       
    def init_market(self):
        for node in self.G.nodes():
            self.G.nodes[node]['S'] = np.random.choice([-1, 1])
            self.G.nodes[node]['C'] = np.random.choice([-1, 1])

        self.M_t = sum(nx.get_node_attributes(self.G, 'S').values())
        self.M_t_values = [self.M_t]
        self.M_t_values_2 = [self.M_t]

    def calculate_neighbor_sum(self, node):
        neighbor_sum = sum(self.G.nodes[neighbor]['S'] for neighbor in self.G.neighbors(node))
        return neighbor_sum

    def update_node(self, node):
        local = self.calculate_neighbor_sum(node)
        h = local - self.G.nodes[node]['C'] * self.alpha * self.M_t / self.N
        p = 1/(1 + np.exp(-self.beta*h))

        #update C
        self.G.nodes[node]['C'] = -self.G.nodes[node]['C'] if self.G.nodes[node]['S'] * self.G.nodes[node]['C'] * self.M_t < 0 else self.G.nodes[node]['C']

        #update S
        rand = random.random()
        if rand < p:
            if self.G.nodes[node]['S'] == -1:
                self.G.nodes[node]['S'] = 1
                self.M_t += 2
        else:
            if self.G.nodes[node]['S'] == 1:
                self.G.nodes[node]['S'] = -1
                self.M_t -= 2
    
    def update_mcs(self, frame):

        for _ in range(self.N):
            rand_node = random.randint(0, self.N-1)
            self.update_node(rand_node)

        self.M_t_values.append(self.M_t)
        #self.M_t_2 = sum(nx.get_node_attributes(self.G, 'S').values())
        #self.M_t_values_2.append(self.M_t_2)
        print(f'MCS = {frame} {self.M_t}')

    def simulate(self, frames = 3000):
        print('debug')
        for t in range(frames):
            #print(f'start loop {t}')
            self.update_mcs(t)
            #print(f'end loop {t}')

    def save_magnetization(self):
        self.M_t_values = pd.Dataframe([self.M_t_values, self.M_t_values_2])
        self.M_t_values.to_csv(SAVE_PATH + f'Test M_t value.csv')
            
if __name__ == '__main__':

    alpha = 10
    beta = 0.8

    print(f'alpha = {alpha}, beta={beta}')
    market = SmallWorldModel(alpha, beta)
    market.init_market()
    market.simulate(frames=3000)
    market.save_magnetization()
    print(market.M_t_values)