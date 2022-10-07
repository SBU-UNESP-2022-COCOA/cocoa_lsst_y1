from cobaya.yaml import yaml_load
from cobaya.post import post
from cobaya.run import run
import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np


post_info = yaml_load(
"""
post:
  suffix: LIN
  remove:
    likelihood:
        lsst_y1.lsst_cosmic_shear:
          path: "./external_modules/data/lsst_y1"
          data_file: LSST_Y1.dataset
          print_datavector: False
          non_linear_emul: 1
          kmax_boltzmann: 5.0
  add:
    likelihood:
        lsst_y1.lsst_cosmic_shear:
          path: "./external_modules/data/lsst_y1"
          data_file: LSST_Y1.dataset
          print_datavector: False
          non_linear_emul: 100
          kmax_boltzmann: 5.0
"""	)


covmat = [[1, -0.85], [-0.85, 1]]
gaussian_info = yaml_load(
"""
likelihood:
  gaussian: "lambda x, y: stats.multivariate_normal.logpdf(
                 [x, y], mean=[0, 0], cov=%s)"

params:
  x:
    prior:
      min: -3
      max:  3
  y:
    prior:
      min: -3
      max:  3

sampler:
  mcmc:
    covmat_params: [x, y]
    covmat: %s
    # Let's impose a strong convergence criterion,
    # to have a fine original sample
    Rminus1_stop: 0.001
""" % (covmat, covmat))



updinfo, sampler = run(gaussian_info)
results = sampler.products()



print(type(updinfo))
print(type(sampler))
print(type(results))


# print(updinfo)
# print(results)