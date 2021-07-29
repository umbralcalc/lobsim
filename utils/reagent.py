import itertools
import numpy as np

class reagentens:
    
    def __init__(self, setup : dict, **kwargs):
        """
        A class for an ensemble of agents which
        can be evolved in time.
        
        Args:
        setup
            A dictionary of setup parameters.
            
        Keywords:
        current_market_state_info
            A dictionary of current market state info from
            the LOB class.
        
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
        
        # Setup storage of past market order volume differences, rate
        # coefficients and memory lengths for the reactionary agents
        self.pastMOdiffmems = 0.0
        self.reactmsk = np.ones(self.setup["Nagents"], dtype=bool)
        self.reactmsk[:self.setup["Nreactagents"]] = False
        self.reactbid = np.ones(self.setup["Nagents"])
        self.reactask = np.ones(self.setup["Nagents"])
        self.MOdiffmemrates = np.random.gamma(
            (
                self.setup["reactratesmean"] ** 2.0
                / np.abs(
                    self.setup["reactratesvar"]
                    - self.setup["reactratesmean"]
                )
            ), 
            self.setup["reactratesmean"] / self.setup["reactratesvar"], 
            size=self.setup["Nreactagents"],
        )
        
        # Draw each agents' initial speculation on best positions 
        # from a unit-mean gamma distribution
        self.logsbids = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=self.setup["Nagents"],
        )
        self.logsasks = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=self.setup["Nagents"],
        )
        self.mogsbids = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=self.setup["Nagents"],
        )
        self.mogsasks = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=self.setup["Nagents"],
        )
        self.mogsbids[:self.setup["Nreactagents"]] = 1.0
        self.mogsasks[:self.setup["Nreactagents"]] = 1.0
        
    def iterate(self, market_state_info : dict):
        """
        Iterate the ensemble a step forward in time by
        asking each agent to make buy-sell-cancel-hold decisions.
        
        Args:
        market_state_info
            A dictionary of current market state info.
            
        """
        
        # Set the reactions of agents in response to their
        # respective memories' of market order volume differences
        self.reactbid[:self.setup["Nreactagents"]] = (
            (self.pastMOdiffmems >= 0.0) * self.setup["reactamp"]
        )
        self.reactask[:self.setup["Nreactagents"]] = (
            (self.pastMOdiffmems <= 0.0) * self.setup["reactamp"]
        )
        
        # Sum over past limit orders by agent
        summembidLOs = np.sum(self.membidLOs, axis=0)
        summemaskLOs = np.sum(self.memaskLOs, axis=0)
        
        # Consistent event rate computations with 
        # agent speculation taken into account
        self.tau = np.random.exponential(1.0 / self.setup["meanHOrate"])
        draws = np.random.uniform(size=(2, self.setup["Nagents"]))
        specs = (
            draws[0] < self.setup["meanspecrate"] / (
                (1.0/self.tau) + self.setup["meanspecrate"]
            )
        )
        nsps = int(np.sum(specs))
        gdraws = np.random.gamma(
            self.setup["heterok"], 
            1.0 / self.setup["heterok"], 
            size=(4, nsps),
        )
        self.logsbids[specs] = gdraws[0]
        self.logsasks[specs] = gdraws[1]
        self.mogsbids[specs] = (
            (1.0 * (self.reactmsk[specs]==False)) 
            + (gdraws[2] * self.reactmsk[specs])
        )
        self.mogsasks[specs] = (
            (1.0 * (self.reactmsk[specs]==False)) 
            + (gdraws[3] * self.reactmsk[specs])
        )
        HOr, LOrb, LOra, MOrb, MOra, COrb, COra = (
            (1.0 / self.tau) * np.ones(self.setup["Nagents"]),
            self.setup["meanLOratebid"] * self.logsbids,
            self.setup["meanLOrateask"] * self.logsasks,
            self.setup["meanMOratebid"] * self.mogsbids * self.reactbid,
            self.setup["meanMOrateask"] * self.mogsasks * self.reactask,
            summembidLOs * self.setup["meanCOratebid"] * self.reactbid,
            summemaskLOs * self.setup["meanCOrateask"] * self.reactask,
        )
        totr = HOr + LOrb + LOra + MOrb + MOra + COrb + COra
        
        # Draw events from the uniform distribution
        # and apply the rejection inequalities to
        # assign the agents' decisions
        evs = draws[1]
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
        agbmos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            MOsb
        ]
        agamos = np.arange(0, self.setup["Nagents"], 1, dtype=int)[
            MOsa
        ]
        nalosdist = np.sum(
            self.memaskLOs[
                midpthigh : market_state_info["askpt"] + 1
            ],
            axis=1,
        )
        nblosdist = np.sum(
            self.membidLOs[
                market_state_info["bidpt"] : midptlow + 1
            ],
            axis=1,
        )
        nalos = np.sum(nalosdist)
        nblos = np.sum(nblosdist)
        alst = np.arange(
            midpthigh, 
            market_state_info["askpt"] + 1, 
            1, 
            dtype=int,
        )
        askmopts = np.asarray(
            list(
                itertools.chain.from_iterable(
                    [
                        [alst[i]] * nalosdist[i] 
                        for i in range(0, len(nalosdist))
                    ]
                )
            )
        )
        blst = np.arange(
            market_state_info["bidpt"], 
            midptlow + 1, 
            1, 
            dtype=int,
        )
        bidmopts = np.asarray(
            list(
                itertools.chain.from_iterable(
                    [
                        [blst[i]] * nblosdist[i] 
                        for i in range(0, len(nblosdist))
                    ]
                )
            )
        )
        alen = (
            nalos * (len(agbmos) > nalos) 
            + len(agbmos) * (len(agbmos) <= nalos)
        )
        blen = (
            nblos * (len(agamos) > nblos) 
            + len(agamos) * (len(agamos) <= nblos)
        )
        if alen > 0:
            self.asks[
                (
                    askmopts[:alen], 
                    np.random.choice(
                        agbmos, 
                        size=alen, 
                        replace=False,
                    ),
                )
            ] -= 1
        if blen > 0:
            self.bids[
                (
                    bidmopts[-blen:], 
                    np.random.choice(
                        agamos, 
                        size=blen, 
                        replace=False,
                    ),
                )
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
        
        # Update the memory of the market order volume differences 
        # for the reactionary agents
        self.pastMOdiffmems = (
            np.sum(self.bids[market_state_info["bidpt"] : midptlow + 1])
            - np.sum(self.asks[midpthigh : market_state_info["askpt"] + 1])
        ) * self.tau + (
            self.pastMOdiffmems * np.exp(-self.MOdiffmemrates * self.tau)
        )