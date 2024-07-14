import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from bornholdt_modulized import FinancialMarketModel

class ErdosRenyiModel(FinancialMarketModel):

    def __init__(self):

        #initialize
        self.N = 1000
        self.p = 0.5
        self.k = 4 #average degree is 4
        self.G = nx.gnp_random_graph(self.N, 2*self.k/(self.N-1))
        
        #self.pos = nx.nx_pydot.pydot_layout(self.G)
        self.alpha = 20
        self.beta = 2

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
            

        

if __name__ == '__main__':
    market = ErdosRenyiModel()
    market.init_market()
    #market.plot_market()
    market.simulate(frames=3000)
    market.plot_magnetization()
    market.plot_returns()
    market.plot_autocorrelation()
    market.plot_return_distribution()
    market.plot_powerlaw()