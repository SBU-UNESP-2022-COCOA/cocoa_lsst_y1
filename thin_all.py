import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np


chains_range1 = np.arange(101, 102) #chains 0-29 in folder1

num_points_thin = 200

analysissettings={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

analysissettings2={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.0',
'range_confidence' : u'0.005'}

for i in chains_range1:
	samples = loadMCSamples('../../gg_split_chains_synthetic/EXAMPLE_MCMC'+str(i),settings=analysissettings)
	samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
	samples.saveAsText('EXAMPLE_MCMC'+str(i)+'_THIN')
	del samples