'''
This file do the experiment of small world network model on different beta, holding alpha constant.
'''
import numpy as np
import pandas as pd
from small_world_modulized import SmallWorldModel
from erdos_renyi_modulized import ErdosRenyiModel

alpha = 20

df = pd.DataFrame()

for beta in [0.2, 0.4, 0.6, 0.8, 1, 1.2]:
    print(f'alpha = {alpha}, beta={beta}')
    market = ErdosRenyiModel(alpha, beta)
    market.init_market()
    market.simulate(frames=3000)
    df[f'{beta}'] = pd.Series(market.M_t_values)

df.to_csv('Erdos Reyni, N=50000, alpha=15, experiment on beta.csv')

print(df.head())