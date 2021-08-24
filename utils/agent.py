import itertools
import numpy as np

class agentens:
    
    def __init__(self, setup : dict, **kwargs):
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
        
        # Draw the initial latent market order volumes and signs
        # from the specified Pareto distribution
        self.latmovs = (
            np.random.pareto(
                self.setup["MOvolpower"],
                size=self.setup["Nagents"],
            ) + 1.0
        ).astype(int)
        self.latmovs[
            self.latmovs > self.setup["MOvolcutoff"]
        ] = self.setup["MOvolcutoff"]
        self.mosigns = (
            1.0 - (
                2.0  
                * np.random.binomial(
                    1, 
                    0.5, 
                    size=self.setup["Nagents"]
                )
            )
        )
        
        # Draw the trader agressiveness factors for limit and
        # market orders from a beta distribution 
        self.moaggrf = np.random.beta(
            self.setup["MOaggrA"], 
            self.setup["MOaggrB"],
            size=self.setup["Nagents"],
        )
        self.loaggrf = np.random.beta(
            self.setup["LOaggrA"], 
            self.setup["LOaggrB"],
            size=self.setup["Nagents"],
        )
        
        # Draw each agents' initial speculation on best positions 
        # from a unit-mean gamma distribution
        gms = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=(6, self.setup["Nagents"]),
        )
        self.logsbids = np.ones(self.setup["Nagents"])
        self.logsasks = np.ones(self.setup["Nagents"])
        self.mogsbids = np.ones(self.setup["Nagents"])
        self.mogsasks = np.ones(self.setup["Nagents"])
        self.cogsbids = np.ones(self.setup["Nagents"])
        self.cogsasks = np.ones(self.setup["Nagents"])
        
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
        
        # Consistent event rate computations with 
        # agent speculation taken into account
        self.tau = np.random.exponential(1.0 / self.setup["meanHOrate"])
        HOr, LOrb, LOra, MOrb, MOra, COrb, COra = (
            (1.0 / self.tau) * np.ones(self.setup["Nagents"]),
            self.setup["meanLOratebid"] * self.logsbids,
            self.setup["meanLOrateask"] * self.logsasks,
            self.setup["meanMOratebid"] * self.mogsbids * (1.0 + self.mosigns),
            self.setup["meanMOrateask"] * self.mogsasks * (1.0 - self.mosigns),
            summembidLOs * self.setup["meanCOratebid"] * self.cogsbids,
            summemaskLOs * self.setup["meanCOrateask"] * self.cogsasks,
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
        midpt = float(
            market_state_info["bidpt"] + market_state_info["askpt"]
        ) / 2.0
        midptlow = int(np.floor(midpt))
        midpthigh = int(np.ceil(midpt))
        LObpts = np.random.choice(
            np.arange(0, midptlow + 1, 1, dtype=int),
            size=self.setup["Nagents"],
            p=(
                dec[:midptlow + 1] / np.sum(dec[:midptlow + 1])
            ),
        )
        LOapts = np.random.choice(
            np.arange(midpthigh, self.setup["Nlattice"], 1, dtype=int),
            size=self.setup["Nagents"],
            p=(
                dec[midpthigh:] / np.sum(dec[midpthigh:])
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
        nMOsa = int(np.sum(MOsa))
        nMOsb = int(np.sum(MOsb))
        nalos = np.sum(self.memaskLOs[market_state_info["askpt"]] > 0)
        nblos = np.sum(self.membidLOs[market_state_info["bidpt"]] > 0)
        alen = nalos * (nMOsb > nalos) + nMOsb * (nMOsb <= nalos)
        blen = nblos * (nMOsa > nblos) + nMOsa * (nMOsa <= nblos)
        if alen > 0:
            agalos = np.random.choice(
                np.arange(0, self.setup["Nagents"], 1, dtype=int),
                size=alen,
                replace=False,
                p=(
                    self.memaskLOs[market_state_info["askpt"]]
                    / np.sum(
                        self.memaskLOs[market_state_info["askpt"]]
                    )
                )
            )
            agamos = np.random.choice(
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[MOsb],
                size=alen,
                replace=False,
            )
            exsize = (
                self.moaggrf[agamos] 
                * self.memaskLOs[market_state_info["askpt"]][agalos]
            ).astype(int)
            exsize[exsize==0] = 1
            exsize[exsize > self.latmovs[agamos]] = self.latmovs[agamos][
                exsize > self.latmovs[agamos]
            ]
            self.asks[market_state_info["askpt"]][agalos] -= exsize
            self.latmovs[agamos] -= exsize
            nchanges = int(np.sum(self.latmovs==0))
            self.mosigns[self.latmovs==0] = (
                1.0 - (
                    2.0 * np.random.binomial(1, 0.5, size=nchanges)
                )
            )
            self.latmovs[self.latmovs==0] = (
                np.random.pareto(
                    self.setup["MOvolpower"],
                    size=nchanges,
                ) + 1.0
            ).astype(int)
            self.latmovs[
                self.latmovs > self.setup["MOvolcutoff"]
            ] = self.setup["MOvolcutoff"]
        if blen > 0:
            agblos = np.random.choice(
                np.arange(0, self.setup["Nagents"], 1, dtype=int),
                size=blen,
                replace=False,
                p=(
                    self.membidLOs[market_state_info["bidpt"]]
                    / np.sum(
                        self.membidLOs[market_state_info["bidpt"]]
                    )
                )
            )
            agbmos = np.random.choice(
                np.arange(0, self.setup["Nagents"], 1, dtype=int)[MOsa],
                size=blen,
                replace=False,
            )
            exsize = (
                self.moaggrf[agbmos]
                * self.membidLOs[market_state_info["bidpt"]][agblos]
            ).astype(int)
            exsize[exsize==0] = 1
            exsize[exsize > self.latmovs[agbmos]] = self.latmovs[agbmos][
                exsize > self.latmovs[agbmos]
            ]
            self.bids[market_state_info["bidpt"]][agblos] -= exsize
            self.latmovs[agbmos] -= exsize
            nchanges = int(np.sum(self.latmovs==0))
            self.mosigns[self.latmovs==0] = (
                1.0 - (
                    2.0 * np.random.binomial(1, 0.5, size=nchanges)
                )
            )
            self.latmovs[self.latmovs==0] = (
                np.random.pareto(
                    self.setup["MOvolpower"],
                    size=nchanges,
                ) + 1.0
            ).astype(int)
            self.latmovs[
                self.latmovs > self.setup["MOvolcutoff"]
            ] = self.setup["MOvolcutoff"]
        
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
    