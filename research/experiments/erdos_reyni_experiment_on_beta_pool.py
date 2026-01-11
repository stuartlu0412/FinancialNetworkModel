'''
This file do the experiment of Erdős-Rényi network model on different beta, holding alpha constant.
'''
import numpy as np
import pandas as pd
import multiprocessing as mp
from src.model.erdos_renyi import ErdosRenyiModel

def run_simulation(Model, alpha, beta):

    print(f'Starting simulation for alpha = {alpha}, beta={beta}.')

    market = Model(alpha, beta)
    market.initialize()
    market.simulate(frames=3000)

    print(f'Simulation for alpha = {alpha}, beta={beta} ended.')
    
    return (alpha, beta, market.M_t_values, market.F_t_values, market.NB_t_values)
    
if __name__ == '__main__':

    # Set the range for alpha and beta
    alpha_range = [10, 20, 30, 40, 50]
    beta_range = [0.2, 0.4, 0.6, 0.8, 1, 1.2]

    param = [(ErdosRenyiModel, alpha, beta) for alpha in alpha_range for beta in beta_range] # Pack alpha and beta into one list of tuples.

    pool = mp.Pool(mp.cpu_count())

    results = pool.starmap(run_simulation, param)

    # Create three dicts to store results
    M_t_data = {}
    F_t_data = {}
    NB_t_data = {}

    for alpha, beta, M_t_values, F_t_values, NB_t_values in results:
        M_t_data[(alpha, beta)] = M_t_values
        F_t_data[(alpha, beta)] = F_t_values
        NB_t_data[(alpha, beta)] = NB_t_values

    print(f'Simulation completed!')

    # Convert the data to pd dataframe
    M_t_df = pd.DataFrame(M_t_data)
    F_t_df = pd.DataFrame(F_t_data)
    NB_t_df = pd.DataFrame(NB_t_data)

    M_t_df.columns = pd.MultiIndex.from_tuples(M_t_df.columns, names=['alpha', 'beta'])
    F_t_df.columns = pd.MultiIndex.from_tuples(F_t_df.columns, names=['alpha', 'beta'])
    NB_t_df.columns = pd.MultiIndex.from_tuples(NB_t_df.columns, names=['alpha', 'beta'])

    M_t_df.to_csv('./results/erdos_renyi/magnetization.csv')
    F_t_df.to_csv('./results/erdos_renyi/fc_ratio.csv')
    NB_t_df.to_csv('./results/erdos_renyi/disorder_bonds.csv')

    print(f'All results saved to ./results/erdos_renyi/')