from cobaya.likelihood import Likelihood
import numpy as np

class EE_bound(Likelihood):
    def initialize(self):
        logA_max = 3.21888
        logA_min = 2.83321
        ns_max   = 1.0
        ns_min   = 0.92
        H0_max   = 73
        H0_min   = 61
        omb_max  = 0.06
        omb_min  = 0.04
        omm_max  = 0.4 
        omm_min  = 0.24 
        w_max    = -0.7
        w_min    = -1.3

        self.EE_bound = np.array(([[logA_min, logA_max],[ns_min, ns_max],[H0_min, H0_max],[omb_min, omb_max], \
                    [omm_min, omm_max],[w_min, w_max]]))

    def get_requirements(self):
        return {
          "logA": None,
          "H0": None,
          "ns": None,
          "omegab": None,
          "omegam": None,
          "w": None,
        }

    def is_within_bounds(self, params, bounds):
        params = np.array(params)
        bounds = np.array(bounds).reshape(-1, 2)
        
        if params.size != bounds.shape[0]:
            raise ValueError("Array and bounds dimensions mismatch")
        
        return np.all((params >= bounds[:, 0]) & (params <= bounds[:, 1]))

    def logp(self, **params_values):
        logA          = self.provider.get_param("logA")
        ns            = self.provider.get_param("ns")
        H0            = self.provider.get_param("H0")
        omegab        = self.provider.get_param("omegab")
        omegam        = self.provider.get_param("omegam")
        w             = self.provider.get_param("w")

        param = np.array([logA, ns, H0, omegab, omegam, w])
        logp  = np.log(self.is_within_bounds(param, self.EE_bound))
        
        return logp