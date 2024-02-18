import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy import stats

N = 100
p = 0.5
alpha = 20
beta = 2

init_position = np.random.random((N, N))
init_strategy = np.random.random((N, N))

S = np.zeros((N, N))
S[init_position >= p] = 1
S[init_position < p] = -1

C = np.zeros((N, N))
C[init_strategy >= p] = 1
C[init_strategy < p] = -1

#Calculate initial energy
M_0 = S.sum()
M_t = M_0

M_t_values = [M_0]

NB = np.zeros((N, N))
for i in range(0, N-1):
    for j in range(0, N-1):
        n = 0
        if S[(i-1)%N][j] != S[i][j]:
            n += 1
        if S[i%N][(j-1)%N] != S[i][j]:
            n += 1
        if S[(i+1)%N][j] != S[i][j]:
            n += 1
        if S[i%N][(j+1)%N] != S[i][j]:
            n += 1
        NB[i][j] = n

NB_t = NB.sum()
 
NB_t_values = [NB_t] #找出鄰居數量的指標


def update(frame):

    global M_t, NB_t, S, C, NB

    def update_NB_for_agent(x, y):
        """Update the disorder count for a changed agent and its neighbors."""
        global S, NB, NB_t
        changes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in changes:
            nx, ny = (x + dx) % N, (y + dy) % N
            if S[x, y] == S[nx, ny]:
                NB[x, y] -= 1
                NB[nx, ny] -= 1
                NB_t -= 2
            else:
                NB[x, y] += 1
                NB[nx, ny] += 1
                NB_t += 2

    for i in range(N*N):    
        rand_x = random.randint(0, N-1)
        rand_y = random.randint(0, N-1)

        local = S[(rand_x-1)%N][(rand_y)%N] \
                + S[(rand_x+1)%N][(rand_y)%N] \
                + S[(rand_x)%N][(rand_y-1)%N] \
                + S[(rand_x)%N][(rand_y+1)%N]

        #local = S[(rand_x-1)][(rand_y-1)] \
        #        + S[(rand_x-1)][(rand_y+1)] \
        #       + S[(rand_x+1)][(rand_y-1)] \
        #        + S[(rand_x+1)][(rand_y+1)]
        
        h = local - C[rand_x][rand_y] * alpha * M_t / (N**2)
        p = 1/(1 + np.exp(-beta*h))

        C[rand_x][rand_y] = -C[rand_x][rand_y] if S[rand_x][rand_y] * C[rand_x][rand_y] * M_t < 0 else C[rand_x][rand_y]
        rand = random.random()

        if rand < p:
            if S[rand_x][rand_y] == -1:
                S[rand_x][rand_y] = 1
                M_t += 2
                
                #calculate NB
                update_NB_for_agent(rand_x, rand_y)

        else:
            if S[rand_x][rand_y] == 1:
                S[rand_x][rand_y] = -1
                M_t -= 2

                #calculate NB
                update_NB_for_agent(rand_x, rand_y)

    M_t_values.append(M_t)
    NB_t_values.append(NB_t)
    
    img.set_array(S)
    plt.title(f'Ising simulation of Financial Market')   
   
    return img,

fig, ax = plt.subplots()
img = ax.imshow(S, animated=True)

ani = animation.FuncAnimation(fig, update, frames=5000, interval=1, blit=True, repeat = False)

plt.show()

M_t_values = pd.Series(M_t_values)
M_t_values.to_csv(f'M_t_{N}.csv')
#returns = np.log(M_t_values / M_t_values.shift(1))
returns = M_t_values - M_t_values.shift(1)
absolute_returns = np.abs(returns)

#plot Magnetization
plt.figure(figsize=(10, 6))
plt.plot(M_t_values, label='M_t')
plt.title(f'Magnetization Over Time, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Frame')
plt.ylabel('Magnetization (M_t)')
plt.legend()
plt.grid(True)
plt.savefig(f'./Result/Magnetization_N={N}.png')
plt.show()

#plot Returns
plt.figure(figsize=(10, 6))
plt.plot(returns, label='M_t')
plt.title(f'Returns Over Time, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Frame')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.savefig(f'./Result/Returns_N={N}.png')
plt.show()

#Plot Return Distribution
excess_kurtosis = stats.kurtosis(returns[np.isfinite(returns)])

plt.hist(returns[np.isfinite(returns)], bins = 100)
plt.title(f'Distribution of Returns: Excess Kurtosis = {excess_kurtosis}, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.savefig(f'./Result/Distribution_N={N}.png')
plt.show()

#Plot Autocorrelation
lags = 100  # Number of lags (you can adjust this based on your requirements)
acf_rt = [pd.Series(returns[np.isfinite(returns)]).autocorr(lag=i) for i in range(lags)]
acf_abs = [pd.Series(absolute_returns[np.isfinite(absolute_returns)]).autocorr(lag=i) for i in range(lags)]

# Plotting the autocorrelation
plt.figure(figsize=(10, 6))
plt.bar(range(lags), acf_rt, color='blue')
plt.title(f'Autocorrelation of Returns, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Lag')
plt.ylabel('ACF Returns')
plt.savefig(f'./Result/ACF_Returns_N={N}.png')
plt.show()

# Plotting the autocorrelation
plt.figure(figsize=(10, 6))
plt.bar(range(lags), acf_abs, color='red')
plt.title(f'Autocorrelation of Absolute Returns, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Lag')
plt.ylabel('ACF Absolute Returns')
plt.savefig(f'./Result/ACF_Absolute_Returns_N={N}.png')
plt.show()

#draw fig 3
sorted_abs_returns = np.sort(absolute_returns)
cumulative_distribution = np.arange(1, len(sorted_abs_returns) + 1) / len(sorted_abs_returns)

# Plotting in a log-log scale similar to Figure 3
plt.figure(figsize=(10, 6))
plt.loglog(sorted_abs_returns, 1 - cumulative_distribution, marker='.', linestyle='none')
plt.title(f'Cumulative Distribution of Absolute Returns (Log-Log Scale), N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Absolute Return (log scale)')
plt.ylabel('Cumulative Probability (log scale)')
plt.grid(True)
plt.show()

# Adjusting the approach to avoid issues with log(0)
# Excluding the tail of the distribution where cumulative probabilities are very close to 1
epsilon = 0.5*1e-1  # Small threshold to exclude the tail
filtered_indices = np.where(np.logical_and(1-cumulative_distribution < 0.5, 1-cumulative_distribution > epsilon))

# Filtered data
filtered_log_returns = np.log(sorted_abs_returns[filtered_indices])
filtered_log_cumulative = np.log((1 - cumulative_distribution)[filtered_indices])

# Linear fit to the filtered log-log data
slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_log_returns[np.logical_and(np.isfinite(filtered_log_returns), np.isfinite(filtered_log_cumulative))], filtered_log_cumulative[np.logical_and(np.isfinite(filtered_log_cumulative), np.isfinite(filtered_log_returns))])

# Plotting the filtered data
plt.figure(figsize=(10, 6))
plt.loglog(sorted_abs_returns, 1 - cumulative_distribution, marker='.', linestyle='none', label='Data')
plt.loglog(sorted_abs_returns[filtered_indices], np.exp(intercept) * sorted_abs_returns[filtered_indices] ** slope, label=f'Fit: exponent = {-slope:.2f}')
plt.title(f'Cumulative Distribution of Absolute Returns (Log-Log Scale) with Power Law Fit, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Absolute Return (log scale)')
plt.ylabel('Cumulative Probability (log scale)')
plt.legend()
plt.grid(True)
plt.savefig(f'./Result/Loglog_N={N}.png')
plt.show()

slope, r_value**2  # Return the slope (power-law exponent) and coefficient of determination

#plot NB
NB_t_values = pd.Series(NB_t_values)
plt.figure(figsize=(10, 6))
plt.plot(NB_t_values, label='NB_t')
plt.title(f'Disorder Over Time, N = {N}, p = {p}, alpha = {alpha}, beta = {beta}')
plt.xlabel('Frame')
plt.ylabel('Disorder (NB_t)')
plt.legend()
plt.grid(True)
plt.savefig(f'./Result/Disorder_N={N}.png')
plt.show()