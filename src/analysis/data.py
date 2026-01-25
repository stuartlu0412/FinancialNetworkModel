import pandas as pd

def load_experiment_result_data(file_path):
    
    data = pd.read_csv(file_path,  header = [0, 1], index_col = 0)
    data.columns = data.columns.set_levels([
        data.columns.levels[0].astype(int),
        data.columns.levels[1].astype(float)
    ])
    return data

def load_single_experiment_data(file_path, params: tuple):
    
    magnetization = load_experiment_result_data(f'../../results/erdos_renyi/magnetization.csv')
    fc_ratio = load_experiment_result_data(f'../../results/erdos_renyi/fc_ratio.csv')
    disorder_bonds = load_experiment_result_data(f'../../results/erdos_renyi/disorder_bonds.csv')

    data = pd.DataFrame()
    data['M_t'] = magnetization[params]
    data['F_t'] = fc_ratio[params]
    data['NB_t'] = disorder_bonds[params]

    return data