def save_magnetization(self):
        self.M_t_values = pd.Series(self.M_t_values)
        self.M_t_values.to_csv(SAVE_PATH + f'M_t_{self.N}_{self.alpha}_{self.beta}.csv')