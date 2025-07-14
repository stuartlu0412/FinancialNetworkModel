'''
Base class for all market models.
All market models should inherit from this class and implement the abstract methods.
'''
import numpy as np
from abc import ABC, abstractmethod

class BaseMarketModel(ABC):

    @abstractmethod
    def init_market(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_market(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def simulate(self):
        raise NotImplementedError("Subclasses must implement this method")