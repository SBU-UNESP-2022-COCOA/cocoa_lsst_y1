####### COLA EMULATOR 1
####### Author: João Victor S. Rebouças, October 2022
####### To use: clone Github repo, import this python script in your work.
import numpy as np
import emulator_funcs as emu
from scipy.interpolate import interp2d
from tensorflow import keras

num_of_points = 400 # How many simulations

###### First of all: load the NN models
nn_models = []
for i in range(len(emu.redshifts)):
    nn_models.append(keras.models.load_model(f'./NN_MODELS/HALOFIT_EMU_1_NHID=2_NEURONS=1024_z={emu.redshifts[i]:.3f}'))
        
###### Loading mins and maxs for rescaling normalized log boosts and PC components
mins, maxs = np.loadtxt('./Data/mins_maxs.txt', unpack=True)
mins_pcs, maxs_pcs = np.loadtxt('./Data/mins_maxs_pcs.txt', unpack=True)

###### Loading average log boosts for the manual PCA reconstruction
averages = np.zeros((len(emu.redshifts), 1200))
for i in range(len(emu.redshifts)):
    averages[i] = np.loadtxt('./Data/Averages/average_{:.2f}.txt'.format(emu.redshifts[i]))

###### Loading PC bases for all redshifts
principal_components = np.zeros((len(emu.redshifts), 6, 1200))
for i in range(len(emu.redshifts)):
    principal_components[i,:,:] = np.loadtxt('./Data/PC_Basis/components_{:.2f}.txt'.format(emu.redshifts[i]))

##### MAIN FUNCTION: get_boost
def get_boost(cosmo_params, ks = np.logspace(-3,2,1200), z = emu.redshifts):
    '''
    Input: cosmo_params, a dictionary of cosmological parameters `As`, `ns`, `Omb`, `Omm`, `h`;
    ks, an array of scales to return
    z, an array of redshifts to return
    Output: Boost B(k, z) = P_NL / P_L for the specified cosmology on the specified grid of k and z.
    Shape: (len(z), len(ks)). Smaller redshifts come first in the array.
    Also output ks.
    Default ks and zs are the COLA outputs.
    '''
    if np.max(z) > np.max(emu.redshifts):
        print('We only emulate boosts up until z = {:.2f}'.format(np.max(emu.redshifts)))
        print('Please, change the input redshift range')
        return -1

    As = cosmo_params['As']
    ns = cosmo_params['ns']
    h  = cosmo_params['h']
    Omega_m = cosmo_params['Omm']
    Omega_b = cosmo_params['Omb']
    
    boost_in_cola_ks_zs = np.zeros((len(emu.redshifts), 1200))
    
    params = np.array([Omega_m, Omega_b, ns, As*10**9, h])
    norm_params = emu.normalize_params(params)
    
    for i in range(len(emu.redshifts)):
        emulated_norm_pcs = nn_models[i](np.array([norm_params])) # NN outputs normalized PC components
        emulated_pcs = emulated_norm_pcs * (maxs_pcs[i] - mins_pcs[i]) + mins_pcs[i]
        emulated_norm_log_boost = averages[i]
        for j in range(6):
            emulated_norm_log_boost += principal_components[i,j,:] * emulated_pcs[0,j] # Manual inverse transformation
        emulated_log_boost = emulated_norm_log_boost * (maxs[i] - mins[i]) + mins[i] # Rescaling log boost
        boost_in_cola_ks_zs[i] = np.exp(emulated_log_boost)
    
    #if np.array_equal(ks, emu.ks_default_precision) and np.array_equal(z, emu.redshifts):
    boost_in_desired_ks_zs = boost_in_cola_ks_zs
    
    return ks, boost_in_desired_ks_zs    