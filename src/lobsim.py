import numpy as np

class lobsim:
    
    def __init__(self, setup : dict, agentens=None):
        """
        A class for simulated limit order books
        which can be evolved in time.
        
        Args:
        setup
            A dictionary of setup parameters.
            
        Keywords:
        agentens
            The class for the ensemble of agents.
        
        """
        self.setup = setup
        self.time = 0.0
        
        # Setup the LOB prices and the market order integer ticks
        self.prices = np.arange(
            self.setup["tickscale"], 
            (float(self.setup["Nlattice"]) * self.setup["tickscale"])
            + self.setup["tickscale"], 
            self.setup["tickscale"],
        )
        self.market_state_info = {
            "bidpt" : self.setup["initbidpricetick"],
            "askpt" : self.setup["initaskpricetick"],
            "midprice" : (
                self.prices[self.setup["initaskpricetick"]] 
                + self.prices[self.setup["initbidpricetick"]]
            ) / 2.0,
            "prices" : self.prices,
            "askptrise" : 0.0,
            "bidptdrop" : 0.0,
        }
        self.bids = np.zeros(self.setup["Nlattice"], dtype=int)
        self.asks = np.zeros(self.setup["Nlattice"], dtype=int)
        self.ae = agentens(
            setup,
            current_market_state_info=self.market_state_info,
        )
        
    def iterate(self):
        """Iterate the book volumes (and the ensemble of agents) 
        a step forward in time."""
        
        # Iterate the agent ensemble
        self.ae.iterate(self.market_state_info)
        
        # Apply the agent orders to the book
        self.time += self.ae.tau
        self.bids += np.sum(self.ae.bids, axis=1)
        self.asks += np.sum(self.ae.asks, axis=1)
        
        # Recalculate the bid-ask spread and mid price
        nza, nzb = np.nonzero(self.asks), np.nonzero(self.bids)
        if len(nza[0]) > 0:
            oldaskpt = self.market_state_info["askpt"]
            self.market_state_info["askpt"] = np.min(nza)
            self.market_state_info["askptrise"] = (
                1.0 * (oldaskpt > self.market_state_info["askpt"])
            )
        if len(nzb[0]) > 0:
            oldbidpt = self.market_state_info["bidpt"]
            self.market_state_info["bidpt"] = np.max(nzb)
            self.market_state_info["bidptdrop"] = (
                1.0 * (oldbidpt < self.market_state_info["bidpt"])
            )
        self.market_state_info["midprice"] = (
            self.prices[self.market_state_info["askpt"]] 
            + self.prices[self.market_state_info["bidpt"]]
        ) / 2.0