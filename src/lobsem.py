import numpy as np
import pandas as pd
from lobsim import lobsim
from mlsolver import mlsolver

class lobsem(lobsim):
    
    def __init__(self, setup : dict, agentens=None):
        """
        A class for emulating microsimulations of
        limit order books.
        
        Args:
        setup
            A dictionary of setup parameters.
            
        Keywords:
        agentens
            The class for the ensemble of agents
            whose behaviours you want to emulate.

        """
        super().__init__(setup, agentens=agentens)

    def train_queues(
        self, 
        burn_in : float = 50.0,
    ):
        """
        Method to train the queue emulation approach
        using Hawkes kernel-based transition probabilities.

        Keywords:
        burn_in
            The burn-in period of the LOB simulation in time.
        
        """

        # Iterate the LOB over time
        tend, t = 5000.0, 0.0
        midps = []
        while t < tend:
            self.iterate()
            t = self.time
            midps.append([t, self.market_state_info["midprice"]])

        # Create time series using the mid price output data
        df = pd.DataFrame(midps, columns=['Time', 'Mid price'])
        df = df[df.Time > burn_in]

