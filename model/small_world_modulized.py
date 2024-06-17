import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from bornholdt_modulized import BornholdtModel

SAVE_PATH = f'/Users/stuartlu/Documents/Complex System/Financial Network Model/results/smallworld'


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

    def plot_market(self):
        #set color for the plot
        value_color_map = {1: 'yellow', -1: 'purple'}
        colors = [value_color_map[self.G.nodes[node]['S']] for node in self.G.nodes()]

        nx.draw(self.G, node_color=colors)
        plt.show()

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
                self.G.nodes[node]['S'] == 1
                self.M_t += 2
        else:
            if self.G.nodes[node]['S'] == 1:
                self.G.nodes[node]['S'] == -1
                self.M_t -= 2

    def plot_frame(self):

        value_color_map = {1: 'yellow', -1: 'purple'}
        colors = [value_color_map[self.G.nodes[node]['S']] for node in self.G.nodes()]
        self.pos = nx.spring_layout(self.G, seed=123)
        nx.draw(self.G, ax=self.ax, node_color=colors, pos=self.pos)
        #nx.draw_networkx_edges(self.G, ax=self.ax, pos = self.pos)

    def update_mcs(self, frame):

        self.ax.clear()

        for _ in range(self.N):
            rand_node = random.randint(0, self.N-1)
            self.update_node(rand_node)

        self.M_t_values.append(self.M_t)
        print(f'MCS = {frame}')

    def simulate_ani(self, frames):
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, self.update_mcs, frames=frames, interval = 1, repeat=False)
        plt.show()

    def simulate(self, frames = 3000, ani=False):
        if ani == True:
            self.simulate_ani(frames=frames)
        else:
            return
            
if __name__ == '__main__':
    '''
    market = ErdosRenyiModel()
    market.init_market()
    #market.plot_market()
    market.simulate(frames=3000, ani=True)
    market.save_magnetization()
    market.plot_magnetization(path = SAVE_PATH)
    market.plot_returns(path = SAVE_PATH)
    market.plot_autocorrelation(path = SAVE_PATH)
    market.plot_return_distribution(path = SAVE_PATH)
    market.plot_powerlaw(path = SAVE_PATH)
    '''
    '''
    for alpha in [10, 15, 20, 25, 30]:
        for beta in [0.8, 0.9, 1, 1.1]:
            print(f'alpha = {alpha}, beta={beta}')
            market = ErdosRenyiModel(alpha, beta)
            market.init_market()
            #market.plot_market()
            market.simulate(frames=3000, ani=True)
            market.save_magnetization()
    '''

    alpha = 10
    beta = 0.8

    print(f'alpha = {alpha}, beta={beta}')
    market = SmallWorldModel(alpha, beta)
    market.init_market()
    #market.plot_market()
    market.simulate(frames=3000, ani=True)
    market.save_magnetization()