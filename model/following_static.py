import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy import stats\
import math
from bornholdt_modulized import BornholdtModel

class StaticFollowerModel(BornholdtModel):

    def __init__(self, N=100, p=0.5, alpha=20, beta=5, lag=100, rho = 0.5):
        self.N = N
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.lag = lag
        self.rho = rho

    def init_market(self):
        init_position = np.random.random((self.N, self.N))
        init_strategy = np.random.random((self.N, self.N))

        self.S = np.zeros((self.N, self.N))
        self.S[init_position >= self.p] = 1
        self.S[init_position < self.p] = -1

        self.C = np.zeros((self.N, self.N))
        self.C[init_strategy >= self.p] = 1
        self.C[init_strategy < self.p] = -1

        self.M_t = self.S.sum()
        self.M_t_values = [self.M_t]

        self.NB = np.zeros((self.N, self.N))
        for i in range(0, self.N-1):
            for j in range(0, self.N-1):
                n = 0
                if self.S[(i-1)%self.N][j] != self.S[i][j]:
                    n += 1
                if self.S[i%self.N][(j-1)%self.N] != self.S[i][j]:
                    n += 1
                if self.S[(i+1)%self.N][j] != self.S[i][j]:
                    n += 1
                if self.S[i%self.N][(j+1)%self.N] != self.S[i][j]:
                    n += 1
                self.NB[i][j] = n

        self.NB_t = self.NB.mean()
        self.NB_t_values = [self.NB_t]

    def update(self, frame):

        def update_NB_for_agent(x, y):
            #Update the disorder count for a changed agent and its neighbors.
                
            changes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in changes:
                nx, ny = (x + dx) % self.N, (y + dy) % self.N
                if self.S[x, y] == self.S[nx, ny]:
                    self.NB[x, y] -= 1
                    self.NB[nx, ny] -= 1
                    self.NB_t -= 2
                else:
                    self.NB[x, y] += 1
                    self.NB[nx, ny] += 1
                    self.NB_t += 2

        for i in range(self.N*self.N):    
            rand_x, rand_y = random.randint(0, self.N-1), random.randint(0, self.N-1)

            
            local = self.S[(rand_x-1)%self.N, rand_y%self.N] \
                    + self.S[(rand_x+1)%self.N, rand_y%self.N] \
                    + self.S[rand_x%self.N, (rand_y-1)%self.N] \
                    + self.S[rand_x%self.N, (rand_y+1)%self.N]
            

            if frame > self.lag:
                max_ = -math.inf
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    R_t_lag = np.array(self.M_t_values[frame-self.lag:frame-1]) - np.array(self.M_t_values[frame-self.lag-1:frame-2]) 
                    win = np.dot(self.S_t[((rand_x+dx)%self.N)*self.N+(rand_y+dy)%self.N, frame-self.lag-1:frame-2], R_t_lag)
                    if win > max_:
                        x, y = rand_x + dx, rand_y + dy
                        max_ = win

                local_max = self.S[x % self.N, y % self.N] * 4

            else:
                local_max = 0
            
            h = self.rho * local_max + (1 - self.rho) * local - self.C[rand_x, rand_y] * self.alpha * self.M_t / (self.N**2)
            p = 1 / (1 + np.exp(-self.beta * h))

            self.C[rand_x][rand_y] = -self.C[rand_x][rand_y] if self.S[rand_x][rand_y] * self.C[rand_x][rand_y] * self.M_t < 0 else self.C[rand_x][rand_y]
            rand = random.random()

            if rand < p:
                if self.S[rand_x][rand_y] == -1:
                    self.S[rand_x][rand_y] = 1
                    self.M_t += 2
                    update_NB_for_agent(rand_x, rand_y)
            else:
                if self.S[rand_x][rand_y] == 1:
                    self.S[rand_x][rand_y] = -1
                    self.M_t -= 2
                    update_NB_for_agent(rand_x, rand_y)

        #record M_t
        self.M_t_values.append(self.M_t)
        self.NB_t_values.append(self.NB_t)

        #record S_t
        for i in range(self.N):
            for j in range(self.N):
                self.S_t[i * self.N + j, frame] = self.S[i][j]

        self.img.set_array(self.S)
        plt.title(f'Ising simulation of Financial Market')
   
        return self.img,

    def simulate(self, frames=3000):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.S, animated=True)

        self.S_t = np.zeros((self.N*self.N, 3000))

        ani = animation.FuncAnimation(fig, self.update, frames=frames, interval=1, blit=True, repeat = False)
        plt.show()
        
        #print(self.M_t_values)
        pd.Series(self.M_t_values).to_csv('M_t.csv')

    def plot_magnetization(self):
        try:
            assert(self.M_t_values)
        except:
            self.M_t_values = pd.read_csv(f'M_t.csv')
        plt.figure(figsize=(10, 6))
        plt.plot(pd.Series(self.M_t_values), label='M_t')
        plt.title('Magnetization Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Magnetization (M_t)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_returns(self):
        try:
            assert(self.M_t_values)
            self.M_t_values = pd.Series(self.M_t_values)
        except:
            self.M_t_values = pd.read_csv(f'M_t.csv')
        #returns = np.log(pd.Series(self.M_t_values).pct_change() + 1)
        returns = np.log(self.M_t_values/self.M_t_values.shift(1))
        plt.figure(figsize=(10, 6))
        plt. plot(returns, label='Returns')
        plt.title('Returns Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_return_distribution(self):
        returns = np.log(self.M_t_values/self.M_t_values.shift(1))
        excess_kurtosis = stats.kurtosis(returns[np.isfinite(returns)])
        plt.hist(returns[np.isfinite(returns)], bins=100)
        plt.title(f'Distribution of Returns: Excess Kurtosis = {excess_kurtosis}')
        plt.show()

    def plot_autocorrelation(self):
        returns = np.log(self.M_t_values/self.M_t_values.shift(1))
        absolute_returns = np.abs(returns)

        lags = 100
        acf_rt = [pd.Series(returns[np.isfinite(returns)]).autocorr(lag=i) for i in range(lags)]
        acf_abs = [pd.Series(absolute_returns[np.isfinite(absolute_returns)]).autocorr(lag=i) for i in range(lags)]

        plt.figure(figsize=(10, 6))
        plt.bar(range(lags), acf_rt, color='blue')
        plt.title('Autocorrelation of Returns')
        plt.xlabel('Lag')
        plt.ylabel('ACF Returns')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(range(lags), acf_abs, color='red')
        plt.title('Autocorrelation of Absolute Returns')
        plt.xlabel('Lag')
        plt.ylabel('ACF Absolute Returns')
        plt.show()

    def plot_powerlaw(self):
        returns = np.log(self.M_t_values/self.M_t_values.shift(1))
        absolute_returns = np.abs(returns)
        sorted_abs_returns = np.sort(absolute_returns)
        cumulative_distribution = np.arange(1, len(sorted_abs_returns) + 1) / len(sorted_abs_returns)

        epsilon = 1e-3  # Small threshold to exclude the tail
        filtered_indices = np.where(1 - cumulative_distribution > epsilon)

        filtered_log_returns = np.log(sorted_abs_returns[filtered_indices])
        filtered_log_cumulative = np.log((1 - cumulative_distribution)[filtered_indices])

        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_log_returns[np.logical_and(np.isfinite(filtered_log_returns), np.isfinite(filtered_log_cumulative))], filtered_log_cumulative[np.logical_and(np.isfinite(filtered_log_cumulative), np.isfinite(filtered_log_returns))])

        plt.figure(figsize=(10, 6))
        plt.loglog(sorted_abs_returns, 1 - cumulative_distribution, marker='.', linestyle='none', label='Data')
        plt.loglog(sorted_abs_returns[filtered_indices], np.exp(intercept) * sorted_abs_returns[filtered_indices] ** slope, label=f'Fit: exponent = {-slope:.2f}')
        plt.title('Cumulative Distribution of Absolute Returns (Log-Log Scale) with Power Law Fit')
        plt.xlabel('Absolute Return (log scale)')
        plt.ylabel('Cumulative Probability (log scale)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_nb_value(self):
        self.NB_t_values = pd.Series(self.NB_t_values)
        plt.figure(figsize=(10, 6))
        plt.plot(self.NB_t_values, label='NB_t')
        plt.title(f'Disorder Over Time, N = {self.N}, p = {self.p}, alpha = {self.alpha}, beta = {self.beta}')
        plt.xlabel('Frame')
        plt.ylabel('Disorder (NB_t)')
        plt.legend()
        plt.grid(True)
        #plt.savefig(f'Disorder_N={self.N}.png')
        plt.show()


if __name__ == '__main__':
    market_model = StaticFollowerModel()
    market_model.init_market()
    market_model.simulate()
    market_model.plot_magnetization()
    market_model.plot_returns()
    market_model.plot_autocorrelation()
    market_model.plot_return_distribution()
    market_model.plot_powerlaw()
    market_model.plot_nb_value()
