'''
This file do the experiment of small world network model on different beta, holding alpha constant.
'''
import numpy as np
import pandas as pd
import multiprocessing as mp
from model.erdos_renyi import ErdosRenyiModel

def run_simulation(alpha, beta, result_queue):

    print(f'Starting simulation for for alpha = {alpha}, beta={beta}.')

    market = ErdosRenyiModel(alpha, beta)
    market.init_market()
    market.simulate(frames=1000)

    print(f'Simulation for for alpha = {alpha}, beta={beta} ended.')
    
    result_queue.put((alpha, beta, market.M_t_values, market.F_t_values)) # Put the result into queue

    print(f'Result put into queue for alpha = {alpha}, beta={beta}.')
    
if __name__ == '__main__':

    # Set the range for alpha and beta
    alpha_range = [10, 20]#[10, 20, 30, 40, 50]#np.arange(10, 50, 10)
    beta_range = [0.2, 0.4]#[0.2, 0.4, 0.6, 0.8, 1, 1.2]#np.arange(0.2, 1.6, 0.2)

    # Create a queue to store results from the processes.
    result_queue = mp.Queue()

    # Create a list to store all the processes.
    processes = []

    for alpha in alpha_range:
        for beta in beta_range:
            print(f'Creating process for alpha = {alpha}, beta={beta}')
            p = mp.Process(target = run_simulation, args = (alpha, beta, result_queue))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
        print(f'Process {p} joined.')
    
    # Create two dicts to st
    M_t_data = {}
    F_t_data = {}

    while not result_queue.empty():
        alpha, beta, M_t_values, F_t_values = result_queue.get()
        M_t_data[(alpha, beta)] = M_t_values
        F_t_data[(alpha, beta)] = F_t_values

    print(f'Simulation completed!')

    # Convert the data to pd dataframe
    M_t_df = pd.DataFrame(M_t_data)
    F_t_df = pd.DataFrame(F_t_data)

    M_t_df.columns = pd.MultiIndex.from_tuples(M_t_df.columns, names = ['alpha', 'beta'])
    F_t_df.columns = pd.MultiIndex.from_tuples(F_t_df.columns, names = ['alpha', 'beta'])

    M_t_df.to_csv('./results/multiprocess_simulation_erdos_reyni/magnetization.csv')
    F_t_df.to_csv('./results/multiprocess_simulation_erdos_reyni/fc_ratio.csv')

    print(f'All result saved')