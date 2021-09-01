import numpy as np
import pandas as pd


class mlensemble:
    def __init__(
        self, drift_func, diffusion_func, integrator,
    ):
        """
        Base class for storage and evolution of a multi-dimensional Langevin ensemble system.
        Args:
        drift_func
            specified drift function A(x,t) for each vector x element
        diffusion_func
            specified diffusion function B(x,t) for the vector x element
        integrator
            specified integrator function for time evolution iteration
        """
        self.dataDict = {}
        self._A = drift_func
        self._B = diffusion_func
        self._Integrator = integrator
        self._tstep = 0
        self._t = 0.0
        self._store_covmat_isTrue = True
        self._firstpass_booleans = None
        self._passed = None
        self._snapshot_rate = None
        self._store = None
        self._x = None

    @property
    def store(self):
        if self._store is None:

            def store_covmat(self):
                pass

            def store_snapshots(self):
                pass

            def store_firstpass(self):
                pass

            if self._store_covmat_isTrue == True:
                self.dataDict["CovMatrix"] = {'t' : [], 'x' : []}

                def store_covmat(self):
                    self.dataDict["CovMatrix"]['t'].append(self.t)
                    self.dataDict["CovMatrix"]['x'].append(np.cov(self.x))

            if self._snapshot_rate is not None:
                self.dataDict["Snapshots"] = {'t' : [], 'x' : []}

                def store_snapshots(self):
                    if (self._tstep % self._snapshot_rate) == 0:
                        self.dataDict["Snapshots"]['t'].append(self.t)
                        self.dataDict["Snapshots"]['x'].append(self.x)

            if self._firstpass_booleans is not None:
                xvals = np.empty_like(self.passed, dtype=float)
                tvals = np.empty_like(self.passed, dtype=float)
                xvals[:], tvals[:] = np.nan, np.nan 
                self.dataDict["FirstPass"] = {'t' : tvals, 'x' : xvals}

                def store_firstpass(self):
                    self.dataDict["FirstPass"]['t'][
                        (~self.passed) & (self._firstpass_booleans)
                    ] = self.t
                    self.dataDict["FirstPass"]['x'][
                        (~self.passed) & (self._firstpass_booleans)
                    ] = np.tensordot(
                        np.ones(self.passed.shape[0]),
                        self.x,
                        axes=0
                    )[(~self.passed) & (self._firstpass_booleans)]
                    self.passed[self._firstpass_booleans] = True

            def store_method(self):
                store_covmat(self)
                store_snapshots(self)
                store_firstpass(self)

            self._store = store_method
        return self._store

    @property
    def passed(self):
        if self._passed is None:
            if self._firstpass_booleans is not None:
                self._passed = np.zeros_like(self._firstpass_booleans, dtype=bool)
        return self._passed

    @staticmethod
    def firstpass_func(prevx: np.ndarray, x: np.ndarray, t: float):
        return None

    def Init(self):
        """Initialise the ensemble system"""

        if self._x is None:
            raise ValueError(
                "No initial condition has been set - use the setup(x0,deltat) method!"
            )
        else:
            self.x = self._x
            self.t = self._t

    def Iterate(self):
        """Iterate the system forward in time by one step and store relevant info"""

        self.prevx = self.x.copy()
        self.x = self._Integrator(self.x, self.t)
        self._tstep += 1
        self._firstpass_booleans = self.firstpass_func(self.prevx, self.x, self.t)
        self.store(self)


class mlsolver:
    def __init__(
        self, drift_func, diffusion_func, solver="IE",
    ):
        """
        Base class for solving multi-dimensional Langevin equations.
        Args:
        drift_func
            specified drift function A(x,t) for each vector x element
        diffusion_func
            specified diffusion function B(x,t) for the vector x element
        Keywords:
        solver
            choice of numerical solver scheme:
                "IE" - Improved Euler scheme, see: https://arxiv.org/abs/1210.0933
        """
        self.A = drift_func
        self.B = diffusion_func
        self.realisations = 1000
        self.dimensions = 2
        self.deltat = None
        self.stratonovich = False
        self.noise_covmat = np.identity(self.dimensions)
        if solver == "IE":
            self.ens = mlensemble(
                drift_func, diffusion_func, self.Improved_Euler_Integrator,
            )
            
    def Improved_Euler_Integrator(self, x : np.ndarray, t : float) -> np.ndarray:
        """Iterate forward one timestep in the Improved Euler scheme"""

        dWt = np.random.multivariate_normal(
            np.zeros(self.dimensions), 
            self.noise_covmat, 
            size=self.realisations,
        ).T
        St = np.random.normal(
            0.0, 1.0, size=(self.dimensions, self.realisations)
        )
        K1 = (self.A(x, t) * self.deltat) + (
            np.sqrt(self.deltat)
            * (dWt - (self.stratonovich==False)*(St / np.abs(St)))
            * self.B(x, t)
        )
        K2 = (self.A(x + K1, t + self.deltat) * self.deltat) + (
            np.sqrt(self.deltat)
            * (dWt + (self.stratonovich==False)*(St / np.abs(St)))
            * self.B(x + K1, t + self.deltat)
        )
        return x + (0.5 * (K1 + K2))

    def setup(self, x0: np.ndarray, deltat: float):
        """Set the initial conditions to the ensemble system and the stepsize"""

        self.ens._x = x0
        self.dimensions = x0.shape[0]
        self.realisations = x0.shape[1]
        self.deltat = deltat

    def run(self, t_period: float):
        """Run the simulation forward in time for a specified period"""

        t = 0
        self.ens.Init()
        while t < t_period:
            self.ens.Iterate()
            t += self.deltat
            self.ens.t = t
            