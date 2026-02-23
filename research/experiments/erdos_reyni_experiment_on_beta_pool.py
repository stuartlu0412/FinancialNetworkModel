'''
This file do the experiment of Erdős-Rényi network model on different beta, holding alpha constant.
'''
import os
import json
import subprocess
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from src.model.erdos_renyi import ErdosRenyiModel

def run_simulation(Model, alpha, beta):

    print(f'Starting simulation for alpha = {alpha}, beta={beta}.')

    market = Model(alpha, beta)
    market.initialize()
    market.simulate(frames=5000)

    print(f'Simulation for alpha = {alpha}, beta={beta} ended.')
    
    return (alpha, beta, market.M_t_values, market.F_t_values, market.NB_t_values)
    
if __name__ == '__main__':

    # Set the range for alpha and beta
    alpha_range = [10, 20, 30, 40, 50]
    beta_range = [0.2, 0.6, 1.2, 1.6, 2, 2.4, 2.6]

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

    # Output experiment results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '../../results/erdos_renyi')
    output_dir = os.path.join(base_dir, f'alpha_beta_sweep_{timestamp}')

    os.makedirs(output_dir, exist_ok=True)
    
    M_t_df.to_csv(f'{output_dir}/magnetization.csv')
    F_t_df.to_csv(f'{output_dir}/fc_ratio.csv')
    NB_t_df.to_csv(f'{output_dir}/disorder_bonds.csv')

    print(f'All results saved to {output_dir}')

    # Output experiment metadata
    git_rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    
    experiment_metadata = {
        "experiment_name": "erdos_renyi_alpha_beta_sweep",
        "timestamp": datetime.now().isoformat(),
        "git_revision": git_rev,
        "model": "ErdosRenyiModel",
        "parameters": {
            "alpha_range": alpha_range,
            "beta_range": beta_range,
            "frames": 5000,
            "total_simulations": len(param),
            "n_workers": mp.cpu_count()
        },
        "output_files": {
            "magnetization": "magnetization.csv",
            "fc_ratio": "fc_ratio.csv",
            "disorder_bonds": "disorder_bonds.csv"
        }
    }
    
    with open(f'{output_dir}/experiment_metadata.json', 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    print(f'Metadata saved to experiment_metadata.json')