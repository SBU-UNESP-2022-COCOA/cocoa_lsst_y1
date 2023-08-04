from cobaya.likelihood import Likelihood
import numpy as np
import os
import torch
### Replaced with load/predict function below; be careful with normalization choicies
#from cobaya.likelihoods.lsst_y1.nn_emulator import NNEmulator 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5


class lsst_emu_3x2_lcdm(Likelihood):
    def initialize(self):


        #self.params            = config.params
        self.probe             = self.probe
        #self.n_walkers         = self.n_emcee_walkers
        self.n_pcas_baryon     = self.n_pcas_baryon
        self.baryon_pcas       = np.loadtxt(self.baryon_pca_file)
        self.emu_type          = self.emu_type
        self.dv_obs            = np.loadtxt(self.data_vector_file)[:,1]
        self.output_dims       = len(self.dv_obs)
        self.dv_len            = self.dv_len
        self.mask              = np.loadtxt(self.mask_file)[:,1].astype(bool)[0:self.dv_len]
        self.cov               = self.get_full_cov(self.cov_file)
        self.masked_inv_cov    = np.linalg.inv(self.cov[0:self.dv_len, 0:self.dv_len][self.mask][:,self.mask]) #firstly do truncation (e.g. only cosmic shear), secondly do masking, finally do inverse
        self.dv_fid            = np.loadtxt(self.data_vector_used_for_training)[:,1]
        self.dv_std            = np.sqrt(np.diagonal(self.cov))
        self.device            = 'cpu'
        self.load_cs(self.emu_file_cs, map_location=torch.device(self.device))
        self.load_2x2(self.emu_file_2x2, map_location=torch.device(self.device))
        if self.emu_file_3x2 is not None:
            self.load_3x2(self.emu_file_3x2, map_location=torch.device(self.device))
        self.shear_calib_mask  = np.load(self.shear_calib_mask) #mask that gives 2 for cosmic shear, and 1 for gg-lensing
        self.source_ntomo      = self.source_ntomo
        if self.probe != 'cosmic_shear':
            self.lens_ntomo = self.lens_ntomo
            self.galaxy_bias_mask = np.load(self.galaxy_bias_mask) # mask that gives 2 for galaxy clustering, and 1 for gg-lensing
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo
        else:
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo
        
        #self.m_shear_prior_std = split_with_comma(self.config_args_emu['shear_calib']['prior_std'])
        
        # NO baryons for now
        #self.config_args_baryons = self.config_args_emu['baryons']
        
        if self.probe!='cosmic_shear':
            #self.config_args_bias  = self.config_args_emu['galaxy_bias']
            self.bias_fid          = self.bias_fid
            self.galaxy_bias_mask  = self.galaxy_bias_mask

        if self.probe!='cosmic_shear':
            self.n_sample_dims    = self.n_dim_2x2 + self.lens_ntomo + self.source_ntomo + self.n_pcas_baryon
        else:
            self.n_sample_dims    = self.n_dim_cs + self.source_ntomo + self.n_pcas_baryon

        #super(lsst_emu_cs_lcdm, self).initialize()

    def get_requirements(self):
        return {
          "logA": None,
          "H0": None,
          "ns": None,
          "omegab": None,
          "omegam": None,
          "omegam_growth": None,
        }
        ### TODO: How to remove calling camb completely?
        ### return a None doesn't solve the problem, and cause problem of "not finding omegab", 
        ### which is weird because omegabh2 to omb is trivial..
        # return {}


    def get_full_cov(self, cov_file):
        print("Getting covariance...")
        full_cov = np.loadtxt(cov_file)
        cov = np.zeros((self.output_dims, self.output_dims))
        cov_scenario = full_cov.shape[1]
        
        for line in full_cov:
            i = int(line[0])
            j = int(line[1])

            if(cov_scenario==3):
                cov_ij = line[2]
            elif(cov_scenario==10):
                cov_g_block  = line[8]
                cov_ng_block = line[9]
                cov_ij = cov_g_block + cov_ng_block

            cov[i,j] = cov_ij
            cov[j,i] = cov_ij

        return cov

    # Get the parameter vector from cobaya
    # TODO: check the order is the same as trained emulator

    def get_theta(self, **params_values):

      theta = np.array([])

      # 6 cosmological parameter for LCDM with gg-split
      logAs = self.provider.get_param("logA")
      ns = self.provider.get_param("ns")
      H0 = self.provider.get_param("H0")
      omegab = self.provider.get_param("omegab")
      omegam = self.provider.get_param("omegam")
      omegam_growth = self.provider.get_param("omegam_growth")
      
      theta = np.append(theta, [logAs, ns, H0, omegab, omegam, omegam_growth ])  #NEED this order for now

      # 7 nuissance parameter emulated
      LSST_DZ_S1 = params_values['LSST_DZ_S1']
      LSST_DZ_S2 = params_values['LSST_DZ_S2']
      LSST_DZ_S3 = params_values['LSST_DZ_S3']
      LSST_DZ_S4 = params_values['LSST_DZ_S4']
      LSST_DZ_S5 = params_values['LSST_DZ_S5']

      LSST_A1_1 = params_values['LSST_A1_1']
      LSST_A1_2 = params_values['LSST_A1_2']



      theta = np.append(theta, [LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4, LSST_DZ_S5, LSST_A1_1, LSST_A1_2])  #NEED this order for now

      # 5 fast parameters don't emulate, no baryons for now
      LSST_M1 = params_values['LSST_M1']
      LSST_M2 = params_values['LSST_M2']
      LSST_M3 = params_values['LSST_M3']
      LSST_M4 = params_values['LSST_M4']
      LSST_M5 = params_values['LSST_M5']

      theta = np.append(theta, [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5])  #NEED this order for now

      if self.probe!='cosmic_shear':
        LSST_DZ_L1 = params_values['LSST_DZ_L1']
        LSST_DZ_L2 = params_values['LSST_DZ_L2']
        LSST_DZ_L3 = params_values['LSST_DZ_L3']
        LSST_DZ_L4 = params_values['LSST_DZ_L4']
        LSST_DZ_L5 = params_values['LSST_DZ_L5']

        LSST_B1_1  = params_values['LSST_B1_1']
        LSST_B1_2  = params_values['LSST_B1_2']
        LSST_B1_3  = params_values['LSST_B1_3']
        LSST_B1_4  = params_values['LSST_B1_4']
        LSST_B1_5  = params_values['LSST_B1_5']

        theta = np.append(theta, [LSST_B1_1, LSST_B1_2, LSST_B1_3, LSST_B1_4, LSST_B1_5])  #NEED this order for now
        theta = np.append(theta, [LSST_DZ_L1, LSST_DZ_L2, LSST_DZ_L3, LSST_DZ_L4, LSST_DZ_L5])  #NEED this order for now
        

      if self.n_pcas_baryon ==4:
        LSST_BARYON_Q1 = params_values['LSST_BARYON_Q1']
        LSST_BARYON_Q2 = params_values['LSST_BARYON_Q2']
        LSST_BARYON_Q3 = params_values['LSST_BARYON_Q3']
        LSST_BARYON_Q4 = params_values['LSST_BARYON_Q4']

        theta = np.append(theta, [LSST_BARYON_Q1, LSST_BARYON_Q2, LSST_BARYON_Q3, LSST_BARYON_Q4])  #NEED this order for now
      if self.n_pcas_baryon ==2:
        LSST_BARYON_Q1 = params_values['LSST_BARYON_Q1']
        LSST_BARYON_Q2 = params_values['LSST_BARYON_Q2']

        theta = np.append(theta, [LSST_BARYON_Q1, LSST_BARYON_Q2])  #NEED this order for now


      # #for 2D check
      # theta = np.array([logAs, ns])
      # print(theta)
      return theta

    # Get the dv from emulator
    def compute_datavector(self, theta):        
        if(self.emu_type=='nn'):
            theta = torch.Tensor(theta)
        elif(self.emu_type=='gp'):
            theta = theta[np.newaxis]

        # Use Cosmic Shear + 2x2pt
        datavector_cs = self.predict_cs(theta[0:self.n_dim_cs])[0]
        datavector_2x2 = self.predict_2x2(theta[0:self.n_dim_2x2])[0]
        datavector = np.append(datavector_cs, datavector_2x2)

        ## Use 3x2 emulator directly (Not Recommended)
        # datavector = self.predict_3x2(theta)[0]
        return datavector

    # add the fast parameter part into the dv
    def get_data_vector_emu(self, theta):
        theta_emu     = np.append(theta[0:self.n_dim_cs], theta[self.n_sample_dims-self.lens_ntomo:]) #get the parameters for emulation
        #print("TESTING0", theta_emu)

        datavector = self.compute_datavector(theta_emu)

        if self.probe!='cosmic_shear':
            #print("TESTING", self.n_sample_dims, self.n_pcas_baryon, self.source_ntomo, self.lens_ntomo )
            bias_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo):
                                  self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo)]
            #print("TESTING2 bias theta", bias_theta, self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo), self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo))
            datavector = self.add_bias(bias_theta, datavector)
            #print("TESTING4",datavector[780:790])
        m_shear_theta = theta[self.n_dim_cs:
                              self.n_dim_cs+self.source_ntomo]
        #print('dv before m_shear',datavector[0:20])
        #print("TESTING3 m_shear_theta", m_shear_theta ,self.n_dim_cs, self.n_dim_cs+self.source_ntomo)
        datavector = self.add_shear_calib(m_shear_theta, datavector)
        #print('dv after m_shear',datavector[0:20])
        if(self.n_pcas_baryon > 0):
            baryon_q   = theta[-self.n_pcas_baryon:]
            # print("TESTING baryon_q=", baryon_q)
            datavector = self.add_baryon_q(baryon_q, datavector)

        return datavector


    def add_bias(self, bias_theta, datavector):
        for i in range(self.lens_ntomo):
            #print("TESTING add bias", self.bias_fid[i], self.galaxy_bias_mask[i][780:790],self.galaxy_bias_mask[i][1500:1510])
            factor = (bias_theta[i] / self.bias_fid[i])**self.galaxy_bias_mask[i]
            #factor = (bias_theta[i])**self.galaxy_bias_mask[i]
            datavector = factor * datavector
        return datavector

    def add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i]
        return datavector

    def add_shear_calib(self, m, datavector):
        for i in range(self.source_ntomo):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:self.dv_len] # for cosmic shear
            datavector = factor * datavector
        return datavector


    def logp(self, **params_values):
        theta = self.get_theta(**params_values)
        model_datavector = self.get_data_vector_emu(theta)
        delta_dv = (model_datavector - self.dv_obs[0:self.dv_len])[self.mask[0:self.dv_len]]

        # ### DEBUGING
        # #print("dv_obs(used for evaluation) = ", self.dv_obs, 'with shape', np.shape(self.dv_obs), self.dv_obs[-10:])
        # #print("emulated_dv = ", model_datavector, "with shape", np.shape(model_datavector), model_datavector[-10:])
        # #print(delta_dv, 'shape: ', np.shape(delta_dv))
        # print("delta_dv / dv_fid = ", delta_dv / self.dv_obs[0:self.dv_len][self.mask[0:self.dv_len]], "lendth is ", len(delta_dv) )
        # ##testing
        # print("testing, saving to test_dv.txt")
        # np.savetxt('test_dv.txt',[self.dv_obs[0:self.dv_len], model_datavector])
        # ###
        # #print("testing.....", self.dv_obs[self.mask] @self.masked_inv_cov @ self.dv_obs[self.mask])
        #print("TESTING fractional difference", ((model_datavector - self.dv_obs)/self.dv_obs)[1550:1560]  )
        #print("TESTING fractional difference", ((model_datavector - self.dv_obs)/self.dv_obs)[780:790]  )
        # ### DEBUG END
        
        logp = -0.5 * delta_dv @ self.masked_inv_cov @ delta_dv  
        return logp

    def load_cs(self, filename, map_location):
        self.trained = True
        self.model_cs = torch.load(filename, map_location)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean_cs  = torch.Tensor(f['X_mean'][:]).float()
            self.X_std_cs   = torch.Tensor(f['X_std'][:]).float()
            self.X_max_cs   = torch.Tensor(f['X_max'][:]).float()
            self.X_min_cs   = torch.Tensor(f['X_min'][:]).float()
            self.dv_fid_cs  = torch.Tensor(f['dv_fid'][:]).float()
            self.dv_std_cs  = torch.Tensor(f['dv_std'][:]).float()
            self.dv_max_cs  = torch.Tensor(f['dv_max'][:]).float()
            self.dv_mean_cs = torch.Tensor(f['dv_mean'][:]).float()
            self.cov_cs     = torch.Tensor(f['cov'][:]).float()
            self.evecs_cs   = torch.Tensor(f['evecs'][:]).float()
            self.evecs_inv_cs  = torch.Tensor(f['evecs_inv'][:]).float()

    def load_2x2(self, filename, map_location):
        self.trained = True
        self.model_2x2 = torch.load(filename, map_location)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean_2x2  = torch.Tensor(f['X_mean'][:]).float()
            self.X_std_2x2   = torch.Tensor(f['X_std'][:]).float()
            self.X_max_2x2   = torch.Tensor(f['X_max'][:]).float()
            self.X_min_2x2   = torch.Tensor(f['X_min'][:]).float()
            self.dv_fid_2x2  = torch.Tensor(f['dv_fid'][:]).float()
            self.dv_std_2x2  = torch.Tensor(f['dv_std'][:]).float()
            self.dv_max_2x2  = torch.Tensor(f['dv_max'][:]).float()
            self.dv_mean_2x2 = torch.Tensor(f['dv_mean'][:]).float()
            self.cov_2x2     = torch.Tensor(f['cov'][:]).float()
            self.evecs_2x2   = torch.Tensor(f['evecs'][:]).float()
            self.evecs_inv_2x2  = torch.Tensor(f['evecs_inv'][:]).float()

    def load_3x2(self, filename, map_location):
        self.trained = True
        self.model_3x2 = torch.load(filename, map_location)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean_3x2  = torch.Tensor(f['X_mean'][:]).float()
            self.X_std_3x2   = torch.Tensor(f['X_std'][:]).float()
            self.X_max_3x2   = torch.Tensor(f['X_max'][:]).float()
            self.X_min_3x2   = torch.Tensor(f['X_min'][:]).float()
            self.dv_fid_3x2  = torch.Tensor(f['dv_fid'][:]).float()
            self.dv_std_3x2  = torch.Tensor(f['dv_std'][:]).float()
            self.dv_max_3x2  = torch.Tensor(f['dv_max'][:]).float()
            self.dv_mean_3x2 = torch.Tensor(f['dv_mean'][:]).float()
            self.cov_3x2     = torch.Tensor(f['cov'][:]).float()
            self.evecs_3x2   = torch.Tensor(f['evecs'][:]).float()
            self.evecs_inv_3x2  = torch.Tensor(f['evecs_inv'][:]).float()

    def predict_cs(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean_cs.clone().detach().to(self.device).float()
            X_std  = self.X_std_cs.clone().detach().to(self.device).float()
            X_max  = self.X_max_cs.clone().detach().to(self.device).float()
            X_min  = self.X_min_cs.clone().detach().to(self.device).float()

            ### mean/std normalization
            # X_norm = (X.to(self.device) - X_mean) / X_std

            ### max/min normalization
            X_norm = (X.to(self.device) - X_min) / (X_max - X_min)
            X_norm = np.reshape(X_norm, (1, len(X_norm)))

            y_pred = self.model_cs.eval()(X_norm).float().cpu() * self.dv_std_cs #normalization

        y_pred = y_pred @ torch.Tensor(np.transpose(self.evecs_cs)) + self.dv_mean_cs #change of basis
        return y_pred.float().numpy()

    def predict_2x2(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean_2x2.clone().detach().to(self.device).float()
            X_std  = self.X_std_2x2.clone().detach().to(self.device).float()
            X_max  = self.X_max_2x2.clone().detach().to(self.device).float()
            X_min  = self.X_min_2x2.clone().detach().to(self.device).float()

            ### mean/std normalization
            # X_norm = (X.to(self.device) - X_mean) / X_std

            ### max/min normalization
            X_norm = (X.to(self.device) - X_min) / (X_max - X_min)
            X_norm = np.reshape(X_norm, (1, len(X_norm)))

            y_pred = self.model_2x2.eval()(X_norm).float().cpu() * self.dv_std_2x2 #normalization

        y_pred = y_pred @ torch.Tensor(np.transpose(self.evecs_2x2)) + self.dv_mean_2x2 #change of basis
        return y_pred.float().numpy()

    def predict_3x2(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean_3x2.clone().detach().to(self.device).float()
            X_std  = self.X_std_3x2.clone().detach().to(self.device).float()
            X_max  = self.X_max_3x2.clone().detach().to(self.device).float()
            X_min  = self.X_min_3x2.clone().detach().to(self.device).float()

            ### mean/std normalization
            # X_norm = (X.to(self.device) - X_mean) / X_std

            ### max/min normalization
            X_norm = (X.to(self.device) - X_min) / (X_max - X_min)
            X_norm = np.reshape(X_norm, (1, len(X_norm)))

            y_pred = self.model_3x2.eval()(X_norm).float().cpu() * self.dv_std_3x2 #normalization

        y_pred = y_pred @ torch.Tensor(np.transpose(self.evecs_3x2)) + self.dv_mean_3x2 #change of basis
        return y_pred.float().numpy()

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias