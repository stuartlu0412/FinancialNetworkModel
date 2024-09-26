import numpy as np
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from model.bornholdt import BornholdtModel

SAVE_PATH = f'/Users/stuartlu/Documents/Complex System/Financial Network Model/results/'


class ErdosRenyiModel():

    def __init__(self, alpha=20, beta=0.8):

        #initialize
        self.seed = 123
        random.seed(self.seed)
        self.N = 50000
        self.p = 0.5
        self.k = 4 #average degree is 4
        self.G = nx.gnp_random_graph(self.N, 2*self.k/(self.N-1), seed = self.seed)
        
        #self.pos = nx.nx_pydot.pydot_layout(self.G)
        self.alpha = alpha
        self.beta = beta
        
        #variables that records all the values for further analysis
        self.M_t = 0 # magnetization
        self.F_t = 0 # number of fundamentalists
        self.M_t_values = []
        self.F_t_values = []
       
    def init_market(self):
        for node in self.G.nodes():
            self.G.nodes[node]['S'] = np.random.choice([-1, 1])
            self.G.nodes[node]['C'] = np.random.choice([-1, 1])

        self.M_t = sum(nx.get_node_attributes(self.G, 'S').values())
        #self.F_t = sum(nx.get_node_attributes(self.G, 'C').values())
        self.F_t = sum([self.G.nodes[node]['C'] for node in self.G.nodes if self.G.nodes[node]['C'] == 1])

        self.M_t_values.append(self.M_t)
        self.F_t_values.append(self.F_t)

    def update_node(self, node):
        local = sum(self.G.nodes[neighbor]['S'] for neighbor in self.G.neighbors(node))
        h = local - self.G.nodes[node]['C'] * self.alpha * self.M_t / self.N
        p = 1/(1 + np.exp(-self.beta*h))

        #update C
        if self.G.nodes[node]['S'] * self.G.nodes[node]['C'] * self.M_t < 0: #Suppose S(t)C(t)M(t) < 0 => C(t+1) = -C(t)
            self.G.nodes[node]['C'] = -self.G.nodes[node]['C']
            if self.G.nodes[node]['C'] > 0:
                self.F_t += 1
            else:
                self.F_t -= 1

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

        if self.M_t > self.N:
            raise Exception("M_t out of bound")

    def update_mcs(self, frame):

        #Randomly choose a node to update
        for _ in range(self.N):
            rand_node = random.randint(0, self.N-1)
            self.update_node(rand_node)

        self.M_t_values.append(self.M_t)
        self.F_t_values.append(self.F_t)
        print(f'MCS = {frame}')

    def simulate(self, frames = 3000):
        print(f'Start simulating with Alpha = {self.alpha}, Beta = {self.beta}')
        for t in range(frames):
            self.update_mcs(t)
            
if __name__ == '__main__':

    alpha = 20
    beta = 2

    print(f'alpha = {alpha}, beta={beta}')
    market = ErdosRenyiModel(alpha, beta)
    market.init_market()
    market.simulate(frames=3000)
    df = pd.concat([pd.Series(market.M_t_values), pd.Series(market.F_t_values)], axis = 1)
    df.to_csv('test_er.csv')