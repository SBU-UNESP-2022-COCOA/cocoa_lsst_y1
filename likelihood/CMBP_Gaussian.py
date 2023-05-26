#KZ: NOT TESTED YET
"""
Fisrt, the covmat_print.py fuction doesn't give consistent results, not sure why.
Second, this likelihood does not give desired results
"""

from cobaya.likelihood import Likelihood
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, uniform, random_correlation

class CMBP_Gaussian(Likelihood):
    def initialize(self):
        covfile = "./projects/lsst_y1/PRIOR_1.covmat"
        covs_dim = 5
        self.input_params = ["logA", "ns", "H0", "omegab", "omegam"]
        temp = pd.read_csv(covfile, header=0).values
        cov  = []
        for i in range(covs_dim):
            line = temp[i][0].split()
            line = [float(j) for j in line]
            cov.append(line[:covs_dim])
        self.cov = np.array(cov)

        self.gaussians = multivariate_normal(mean=self.mean, cov=cov)


        self.weights = 1

    def get_requirements(self):
        return {
          # "logA": None,
          # "H0": None,
          # "ns": None,
          # "omegab": None,
          # "omegam": None,
          # "w": None,
        }

    def logp(self, **params_values):

        x = np.array([params_values[p] for p in self.input_params])

        # Compute the likelihood and return
        return self.gaussians.logpdf(x)