import numpy as np

class SFagentens:
    
    def __init__(self, setup : dict):
        """
        A class for an ensemble of agents which
        can be evolved in time.
        
        Args:
        setup
            A dictionary of setup parameters.
        
        """
        self.setup = setup
        
        # Setup the bid and ask decision properties
        self.bids = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]), 
            dtype=int,
        )
        self.asks = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]), 
            dtype=int,
        )
        
        # Keep a memory of all outstanding limit orders,
        # their volumes and which agent they are associated to
        self.membidLOs = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]),
            dtype=int,
        )
        self.memaskLOs = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]),
            dtype=int,
        )
        
    def iterate(self, market_state_info : dict):
        """
        Iterate the ensemble a step forward in time by
        asking each agent to make buy-sell-cancel-hold decisions.
        
        Args:
        market_state_info
            A dictionary of current market state info.
            
        """
        # Sum over past limit orders by agent
        summembidLOs = np.sum(self.membidLOs, axis=0)
        summemaskLOs = np.sum(self.memaskLOs, axis=0)
        
        # Consistent event rate computations
        self.tau = np.random.exponential(1.0 / self.setup["HOrate"])
        HOr, LOr, MOr, COr = (
            self.tau * np.ones(self.setup["Nagents"]),
            self.setup["LOrateperagent"] * np.ones(
                self.setup["Nagents"]
            ),
            self.setup["MOrateperagent"] * np.ones(
                self.setup["Nagents"]
            ),
            (
                (summembidLOs + summemaskLOs)
                * self.setup["COrateperagent"]
            ),
        )
        totr = HOr + LOr + MOr + COr
        
        # Draw events from the uniform distribution
        # and apply the rejection inequalities to
        # assign the agents' decisions
        evs = np.random.uniform(size=self.setup["Nagents"])
        LOs = (evs < LOr / totr)
        MOs = (LOr / totr <= evs) & (evs < (LOr + MOr) / totr)
        COs = (
            ((LOr + MOr) / totr <= evs)
            & (evs < (LOr + MOr + COr) / totr)
        )
        
        # Decide on the prices for both the limit orders
        # and the order cancellations
        boa = (
            np.random.binomial(1, 0.5, size=self.setup["Nagents"]) 
            == 1
        )
        prs = np.random.uniform(
            size=(self.setup["Nlattice"], self.setup["Nagents"])
        )
        dec = np.exp(
            - np.abs(
                market_state_info["midprice"]
                - market_state_info["prices"]
            ) * self.setup["LOdecay"]
        )
        LObpts = np.random.choice(
            np.arange(
                0, 
                market_state_info["bidpt"] + 1, 
                1, 
                dtype=int,
            ),
            size=self.setup["Nagents"],
            p=(
                dec[:market_state_info["bidpt"] + 1]
                / np.sum(dec[:market_state_info["bidpt"] + 1])
            ),
        )
        LOapts = np.random.choice(
            np.arange(
                market_state_info["askpt"], 
                self.setup["Nlattice"], 
                1, 
                dtype=int,
            ),
            size=self.setup["Nagents"],
            p=(
                dec[market_state_info["askpt"]:]
                / np.sum(dec[market_state_info["askpt"]:])
            ),
        )
        CObpts = np.argmax(
            (
                prs
                * ((self.membidLOs + self.memaskLOs) > 0)
                * boa
            ), 
            axis=0,
        )
        COapts = np.argmax(
            (
                prs
                * ((self.membidLOs + self.memaskLOs) > 0)
                * (boa==False)
            ), 
            axis=0,
        )
        
        # Pass the limit-order and market-order decisions 
        # on to the output properties while leaving some 
        # market orders unfulfilled if there are too many 
        # of them
        self.bids[:], self.asks[:] = 0, 0
        self.bids[
            (
                LObpts[boa & LOs], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    boa & LOs
                ],
            )
        ] += 1
        self.asks[
            (
                LOapts[(boa==False) & LOs], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    (boa==False) & LOs
                ],
            )
        ] += 1
        agbmos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            boa & MOs
        ]
        agamos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            (boa==False) & MOs
        ]
        nalos = np.sum(
            (self.memaskLOs + self.asks)[market_state_info["askpt"]]
        )
        nblos = np.sum(
            (self.membidLOs + self.bids)[market_state_info["bidpt"]]
        )
        self.asks[
            market_state_info["askpt"], 
            np.random.choice(
                agbmos, 
                size=(
                    nalos * (len(agbmos) > nalos) 
                    + len(agbmos) * (len(agbmos) <= nalos)
                ), 
                replace=False,
            ),
        ] -= 1
        self.bids[
            market_state_info["bidpt"], 
            np.random.choice(
                agamos, 
                size=(
                    nblos * (len(agamos) > nblos) 
                    + len(agamos) * (len(agamos) <= nblos)
                ), 
                replace=False,
            ),
        ] -= 1
        
        # Pass the cancel-order decisions on to the output
        # properties if they haven't already been fulfilled
        nbidlos = np.sum(self.membidLOs + self.bids, axis=1)
        nasklos = np.sum(self.memaskLOs + self.asks, axis=1)
        cbids = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]), 
        )
        casks = np.zeros(
            (self.setup["Nlattice"], self.setup["Nagents"]), 
        )
        cbids[
            (
                CObpts[COs & boa], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    COs & boa
                ],
            )
        ] += np.random.uniform(size=len(CObpts[COs & boa]))
        casks[
            (
                COapts[COs & (boa==False)], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    COs & (boa==False)
                ],
            )
        ] += np.random.uniform(size=len(COapts[COs & (boa==False)]))
        cbidssinds = np.argsort(cbids, axis=1)
        caskssinds = np.argsort(casks, axis=1)
        cbput = 0 + 1 * (
            np.tensordot(
                self.setup["Nlattice"] - np.minimum(
                    np.sum(cbids > 0, axis=1), 
                    nbidlos,
                ),
                np.ones(self.setup["Nagents"]),
                axes=0,
            ) <= np.tensordot(
                np.ones(self.setup["Nlattice"]),
                np.arange(0, self.setup["Nagents"], 1),
                axes=0,
            )
        )
        caput = 0 + 1 * (
            np.tensordot(
                self.setup["Nlattice"] - np.minimum(
                    np.sum(casks > 0, axis=1), 
                    nasklos,
                ),
                np.ones(self.setup["Nagents"]),
                axes=0,
            ) <= np.tensordot(
                np.ones(self.setup["Nlattice"]),
                np.arange(0, self.setup["Nagents"], 1),
                axes=0,
            )
        )
        np.put_along_axis(cbids, cbidssinds, cbput, axis=1)
        np.put_along_axis(casks, caskssinds, caput, axis=1)
        self.bids -= cbids.astype(int)
        self.asks -= casks.astype(int)
        
        # Update the agent-specific limit order memory
        self.membidLOs += self.bids
        self.memaskLOs += self.asks