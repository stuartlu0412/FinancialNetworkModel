'''
This file do the experiment of small world network model on different beta, holding alpha constant.
'''
import numpy as np
import pandas as pd
from model.small_world import SmallWorldModel

alpha = 10

df = pd.DataFrame()

for beta in [0.2, 0.4, 0.6, 0.8, 1, 1.2]:
    print(f'alpha = {alpha}, beta={beta}')
    market = SmallWorldModel(alpha, beta)
    market.init_market()
    market.simulate(frames=3000)
    df[f'{beta}'] = pd.Series(market.M_t_values)

df.to_csv('Small World, N=50000, alpha=10, experiment on beta')

print(df.head())