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
        
        # Setup storage for exponential Hawkes kernel integral
        self.hawkesintbids = 0.0
        self.hawkesintasks = 0.0
        
        # Setup the each traders' exogeneous vs endogeneous 
        # behaviour ratio and Hawkes kernel power if these
        # have been set, else the default purely exogeneous
        if "rbehaviours" in self.setup:
            self.ris = self.setup["rbehaviours"]
        else:
            self.ris = np.ones(self.setup["Nagents"])
        if "Hawkespow" in self.setup:
            self.hawkespow = self.setup["Hawkespow"]
        else:
            self.hawkespow = np.zeros(self.setup["Nagents"])
        
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
        
        # Draw each agents' preferred relative trading rate
        # from a unit-mean gamma distribution
        self.gs = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=self.setup["Nagents"],
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
        lb, la, mb, ma, cb, ca = market_state_info["LOMOCOexogstate"]
        HOr, LOrb, LOra, MOrb, MOra, COrb, COra = (
            self.tau * np.ones(self.setup["Nagents"]),
            (
                lb * self.gs * self.ris
            ) + ((1.0 - self.ris) * self.hawkesintbids),
            (
                la * self.gs * self.ris
            ) + ((1.0 - self.ris) * self.hawkesintasks),
            (
                mb * self.gs * self.ris
            ) + ((1.0 - self.ris) * self.hawkesintbids),
            (
                ma * self.gs * self.ris
            ) + ((1.0 - self.ris) * self.hawkesintasks),
            (summembidLOs * cb * self.gs),
            (summemaskLOs * ca * self.gs),
        )
        totr = HOr + LOrb + LOra + MOrb + MOra + COrb + COra
        
        # Draw events from the uniform distribution
        # and apply the rejection inequalities to
        # assign the agents' decisions
        evs = np.random.uniform(size=self.setup["Nagents"])
        LOsb = (evs < LOrb / totr)
        LOsa = (LOrb / totr <= evs) * (evs < (LOra + LOrb) / totr)
        MOsb = (
            ((LOra + LOrb) / totr <= evs) 
            & (evs < (LOra + LOrb + MOrb) / totr)
        )
        MOsa = (
            ((LOra + LOrb + MOrb) / totr <= evs) 
            & (evs < (LOra + LOrb + MOrb + MOra) / totr)
        )
        COsb = (
            ((LOra + LOrb + MOrb + MOra) / totr <= evs) 
            & (evs < (LOra + LOrb + MOrb + MOra + COrb) / totr)
        )
        COsa = (
            ((LOra + LOrb + MOrb + MOra + COrb) / totr <= evs) 
            & (
                evs < (
                    LOra + LOrb + MOrb + MOra + COrb + COra
                ) / totr
            )
        )
        
        # Decide on the prices for both the limit orders
        # and the order cancellations
        boa = (LOsb | MOsb | COsb)
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
                LObpts[LOsb], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    LOsb
                ],
            )
        ] += 1
        self.asks[
            (
                LOapts[LOsa], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    LOsa
                ],
            )
        ] += 1
        agbmos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            MOsb
        ]
        agamos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            MOsa
        ]
        nalos = np.sum(self.memaskLOs[market_state_info["askpt"]])
        nblos = np.sum(self.membidLOs[market_state_info["bidpt"]])
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
                CObpts[COsb], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    COsb
                ],
            )
        ] += np.random.uniform(size=len(CObpts[COsb]))
        casks[
            (
                COapts[COsa], 
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[
                    COsa
                ],
            )
        ] += np.random.uniform(size=len(COapts[COsa]))
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
        
        # Update the Hawkes kernel integrals
        self.hawkesintbids = (
            (self.hawkesintbids * np.exp(-self.hawkespow * self.tau)) + 
            np.sum(self.bids * self.tau)
        )
        self.hawkesintasks = (
            (self.hawkesintasks * np.exp(-self.hawkespow * self.tau)) + 
            np.sum(self.asks * self.tau)
        )