'''
Base class for all market models.
All market models should inherit from this class and implement the abstract methods.
'''
from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
import json
import subprocess

class BaseMarketModel(ABC):
    """
    Abstract base for all market models.
    All subclasses must implement:
      • initialize(): set up initial state, seeds, time series lists  
      • step():        one Monte-Carlo sweep (N random updates)  
    Base provides:
      • run(n_steps):  calls initialize() then step() in a loop  
                      and returns a DataFrame of the recorded time series  
      • _snapshot_params(): writes out a JSON of params + git-hash for reproducibility  
    """
    def __init__(self, seed: int = 123):
        self.seed = seed

    def _snapshot_params(self, output_dir: str):
        """Write out self.__dict__ + git commit hash to output_dir/params.json"""
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip()
        payload = dict(self.__dict__, git_revision=git_rev)
        # only include simple constructor params, not arrays or large lists
        primitive_params = {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, (int, float, str, bool))
        }
        payload = {
            "git_revision": git_rev,
            "params": primitive_params
        }
        with open(f"{output_dir}/params.json", "w") as f:
            json.dump(payload, f, indent=2)
    

    @abstractmethod
    def initialize(self) -> None:
        """Set up all state variables and time‐series lists (e.g. M_t_values)."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def step(self) -> None:
        """Perform exactly one MC sweep (i.e. N random updates)."""
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, n_steps: int, output_dir: str = None) -> pd.DataFrame:
        """
        1) seed RNGs  
        2) initialize()  
        3) step() n_steps times  
        4) return DataFrame of recorded series  
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.initialize()
        if output_dir:
            self._snapshot_params(output_dir)

        for t in range(n_steps):
            self.step()

        # assume subclass has built e.g. self.M_t_values, self.F_t_values
        df = pd.DataFrame({
            "M_t": self.M_t_values,
            "F_t": getattr(self, "F_t_values", None)
        })
        if output_dir:
            df.to_csv(f"{output_dir}/timeseries.csv", index=False)
        return df