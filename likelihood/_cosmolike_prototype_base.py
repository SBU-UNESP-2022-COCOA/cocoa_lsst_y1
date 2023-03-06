# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import CubicSpline, interp1d
import sys
import time
from scipy.integrate import odeint
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy import optimize

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

#Growth-Split (gs) BEGINS:
import euclidemu2
import math
#Growth-Split (gs) ENDS:

import cosmolike_lsst_y1_interface as ci

# default is best fit LCDM - just need to be an ok Cosmology
default_omega_matter = 0.315
default_hubble = 74.03
default_omega_nu_h2 = 0.0

default_z = np.array([
  0.,          0.11101010,  0.22202020,  0.33303030,  0.44404040,  0.55505051,
  0.66606061,  0.77707071,  0.88808081,  0.99909091,  1.11010101,  1.22111111,
  1.33212121,  1.44313131,  1.55414141,  1.66515152,  1.77616162,  1.88717172,
  1.99818182,  2.10919192,  2.22020202,  2.33121212,  2.44222222,  2.55323232,
  2.66424242,  2.77525253,  2.88626263,  2.99727273,  3.10828283,  3.21929293,
  3.33030303,  3.44131313,  3.55232323,  3.66333333,  3.77434343,  3.88535354,
  3.99636364,  4.10737374,  4.21838384,  4.32939394,  4.44040404,  4.55141414,
  4.66242424,  4.77343434,  4.88444444,  4.99545455,  5.10646465,  5.21747475,
  5.32848485,  5.43949495,  5.55050505,  5.66151515,  5.77252525,  5.88353535,
  5.99454545,  6.10555556,  6.21656566,  6.32757576,  6.43858586,  6.54959596,
  6.66060606,  6.77161616,  6.88262626,  6.99363636,  7.10464646,  7.21565657,
  7.32666667,  7.43767677,  7.54868687,  7.65969697,  7.77070707,  7.88171717,
  7.99272727,  8.10373737,  8.21474747,  8.32575758,  8.43676768,  8.54777778,
  8.65878788,  8.76979798,  8.88080808,  8.99181818,  9.10282828,  9.21383838,
  9.32484848,  9.43585859,  9.54686869,  9.65787879,  9.76888889,  9.87989899,
  9.99090909, 10.10191919, 10.21292929, 10.32393939, 10.43494949, 10.54595960,
  10.6569697, 10.76797980, 10.87898990, 10.99000000
  ])

default_chi = np.array([
   0.0,439.4981441901221,858.0416237527756,1254.8408009609223,1629.709749743714,
   1982.9621087888709,2315.2896285308143,2627.6450104613505,2921.140613926014,
   3196.968260667916,3456.339782258587,3700.4456271142944,3930.4279102286982,
   4147.3643438650115,4352.259985452185,4546.044385041486,4729.5721949378385,
   4903.626171493769,5068.921287405417,5226.109582520614,5375.785225614292,
   5518.489536109791,5654.715801869407,5784.913800415938,5909.493978237531,
   6028.8312839788305,6143.26859641929,6253.119923208842,6358.673177750374,
   6460.192721741891,6557.921633986121,6652.083741920545,6742.885440028272,
   6830.517317562262,6915.1556161061435,6996.963542799694,7076.092412212595,
   7152.682741675389,7226.865166025485,7298.761294688355,7368.4844768438925,
   7436.140491426934,7501.828170160967,7565.63996211106,7627.6624396908255,
   7687.976770284773,7746.659112922598,7803.781032248162,7859.409823733494,
   7913.608835484643,7966.437756844108,8017.952882055976,8068.207351447877,
   8117.251372316519,8165.132421467811,8211.895435317594,8257.582964035817,
   8302.23535405978,8345.890875994064,8388.585862443648,8430.354831318795,
   8471.230599781875,8511.244389653995,8550.425925019172,8588.803522692133,
   8626.404179504872,8663.253636772872,8699.376473063612,8734.796157982593,
   8769.535118356289,8803.614796722586,8837.055705886985,8869.877479851164,
   8902.098921393474,8933.738046556602,8964.812129049888,8995.337728085331,
   9025.330741701371,9054.80642977566,9083.779449000327,9112.263883075733,
   9140.273271085562,9167.820634180674,9194.918500688977,9221.578929759315,
   9247.813535983449,9273.63350098463,9299.04960740926,9324.072248139235,
   9348.711446338273,9372.976872144722,9396.877858455327,9420.423415857691,
   9443.622246764959,9466.482758802447,9489.013079507331,9511.221060266003,
   9533.114299897283,9554.700147330219,9575.985713926271
 ])

class _cosmolike_prototype_base(DataSetLikelihood):
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def initialize(self, probe):

    # ------------------------------------------------------------------------
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))

    self.probe = probe

    self.IA_type = ini.int('IA_model')

    self.data_vector_file = ini.relativeFileName('data_file')

    self.cov_file = ini.relativeFileName('cov_file')

    self.mask_file = ini.relativeFileName('mask_file')

    self.lens_file = ini.relativeFileName('nz_lens_file')

    self.source_file = ini.relativeFileName('nz_source_file')

    self.ggl_olap_cut = ini.float("lensing_overlap_cut")

    self.lens_ntomo = ini.int("lens_ntomo") #5

    self.source_ntomo = ini.int("source_ntomo") #4

    self.ntheta = ini.int("n_theta")

    self.theta_min_arcmin = ini.float("theta_min_arcmin")

    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    self.force_cache_false = False

    # ------------------------------------------------------------------------
    self.z_interp_1D = np.linspace(0,2.0,1000)
    self.z_interp_1D = np.concatenate((self.z_interp_1D,
      np.linspace(2.0,10.1,200)),axis=0)
    self.z_interp_1D = np.concatenate((self.z_interp_1D,
      np.linspace(1080,2000,20)),axis=0) #CMB 6x2pt g_CMB (possible in the future)
    self.z_interp_1D[0] = 0

    #Growth-Split (gs) BEGINS:
    self.z_interp_2D = np.linspace(0,2.0,95)
    self.z_interp_2D = np.concatenate((self.z_interp_2D, np.linspace(2.0,10.0,5)),axis=0)
    self.z_interp_2D[0] = 0
    #Growth-Split (gs) ENDS:
    self.len_z_interp_2D = len(self.z_interp_2D)
    #Growth-Split (gs) BEGINS:
    self.len_log10k_interp_2D = 1200
    self.log10k_interp_2D = np.linspace(-4.2,2.0,self.len_log10k_interp_2D)
    #Growth-Split (gs) ENDS:

    # Cobaya wants k in 1/Mpc
    self.k_interp_2D = np.power(10.0,self.log10k_interp_2D)
    self.len_k_interp_2D = len(self.k_interp_2D)
    self.len_pkz_interp_2D = self.len_log10k_interp_2D*self.len_z_interp_2D
    self.extrap_kmax = 2.5e2 * self.accuracyboost

    # ------------------------------------------------------------------------

    ci.initial_setup()
    ci.init_accuracy_boost(self.accuracyboost, self.samplingboost, self.integration_accuracy)

    ci.init_probes(possible_probes=self.probe)

    ci.init_binning(self.ntheta, self.theta_min_arcmin, self.theta_max_arcmin)

    ci.init_cosmo_runmode(is_linear=False)

    # to set lens tomo bins, we need a default \chi(z)
    ci.set_cosmological_parameters(omega_matter = default_omega_matter, hubble = default_hubble, 
      is_cached = False)

    # convert chi to Mpc/h
    ci.init_distances(default_z, default_chi*default_hubble/100.0)

    ci.init_IA(self.IA_type)

    ci.init_source_sample(self.source_file, self.source_ntomo)

    ci.init_lens_sample(self.lens_file, self.lens_ntomo, self.ggl_olap_cut)

    ci.init_size_data_vector()

    ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

    # FOR ALLOWED OPTIONS FOR `which_baryonic_simulations`, SEE BARYONS.C
    # FUNCTION `void init_baryons(char* scenario)`. SIMS INCLUDE
    # TNG100, HzAGN, mb2, illustris, eagle, owls_AGN_T80, owls_AGN_T85,
    # owls_AGN_T87, BAHAMAS_T76, BAHAMAS_T78, BAHAMAS_T80
    ci.init_baryons_contamination(self.use_baryonic_simulations_for_dv_contamination,
      self.which_baryonic_simulations_for_dv_contamination)

    if self.create_baryon_pca:
      ci.init_baryon_pca_scenarios(self.baryon_pca_select_simulations)
      self.use_baryon_pca = False
    else:
      if ini.string('baryon_pca_file', default=''):
        baryon_pca_file = ini.relativeFileName('baryon_pca_file')
        self.baryon_pcs = np.loadtxt(baryon_pca_file)
        self.log.info('use_baryon_pca = True')
        self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
        self.use_baryon_pca = True
      else:
        self.log.info('use_baryon_pca = False')
        self.use_baryon_pca = False

    self.baryon_pcs_qs = np.zeros(4)
    # ------------------------------------------------------------------------

    self.do_cache_lnPL = np.zeros(self.len_log10k_interp_2D*self.len_z_interp_2D)

    self.do_cache_lnPNL = np.zeros(self.len_log10k_interp_2D*self.len_z_interp_2D)

    self.do_cache_chi = np.zeros(len(self.z_interp_1D))

    self.do_cache_cosmo = np.zeros(2)

    # ------------------------------------------------------------------------
    if self.non_linear_emul in[1,3,4,5,6,7]:
      self.emulator = ee2=euclidemu2.PyEuclidEmulator()
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    return {
      "H0": None,
      "omegab": None,
      "omegam": None,
      "omegam_growth": None,
      "omegan2": None,
      "w": None,
      "w_growth": None,
      "zstar":None,
      "Pk_interpolator": {
        "z": self.z_interp_2D,
        "k_max": self.kmax_boltzmann * self.accuracyboost,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
      },
      "comoving_radial_distance": {
        "z": self.z_interp_1D
      # Get comoving radial distance from us to redshift z in Mpc.
      },
      "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
      }
    }

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cache_alert(self, chi, lnPL, lnPNL):
    cache_alert_1 = np.array_equal(self.do_cache_chi, chi)

    cache_alert_2 = np.array_equal(self.do_cache_lnPL, lnPL)

    cache_alert_3 = np.array_equal(self.do_cache_lnPNL, lnPNL)

    cache_alert_4 = np.array_equal(
      self.do_cache_cosmo,
      np.array([
        self.provider.get_param("omegam"),
        self.provider.get_param("H0")
      ])
    )

    return cache_alert_1 and cache_alert_2 and cache_alert_3 and cache_alert_4 and not self.force_cache_false

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0

    OM_GROWTH = self.provider.get_param("omegam_growth")
    if OM_GROWTH < -50:
      OM_GROWTH = self.provider.get_param("omegam")

    W_GROWTH = self.provider.get_param("w_growth")
    if W_GROWTH < -50:
      W_GROWTH = self.provider.get_param("w")

    # Compute linear matter power spectrum
    PKL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear=False, 
      extrap_kmax = self.extrap_kmax)

    # Compute non-linear matter power spectrum
    PKNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear=True, 
        extrap_kmax = self.extrap_kmax)

    #Growth-Split (gs) BEGINS:
    zgs  = np.flip(np.concatenate((self.z_interp_2D, np.linspace(10.1,1000,3000)),axis=0)) 
    
    def G_GROWTH_ODE(y, N):
      OMG   = OM_GROWTH
      WG    = W_GROWTH
      f_nu  = self.provider.get_param("omegan2")/(h**2)/OM_GROWTH
      z     = 1.0/np.exp(N) - 1.0 
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3*(1.0 + WG)) # H^2(z)
      OMG_A = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) / H2                       # Omega_m(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG)) / H2                         # Omega_DE(z)
      HPH   =  -1.5*OMG_A -1.5*OLG_A*(1.0 + WG)                            # dlnH/dlna
      return [y[1], - (4 + HPH)*y[1] - (3.0 + HPH - 1.5 * OMG_A * (1-f_nu))*y[0]]     # [dG/dlna, d^2G/dlna^2]

    def G_GEO_ODE(y, N):
      OMG   = self.provider.get_param("omegam")
      WG    = self.provider.get_param("w")
      f_nu  = self.provider.get_param("omegan2")/(h**2)/self.provider.get_param("omegam")
      z     = 1.0/np.exp(N) - 1.0 
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3*(1.0 + WG)) # H^2(z)
      OMG_A = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) / H2                       # Omega_m(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG)) / H2                         # Omega_DE(z)
      HPH   =  -1.5*OMG_A -1.5*OLG_A*(1.0 + WG)                            # dlnH/dlna
      return [y[1], - (4 + HPH)*y[1] - (3.0 + HPH - 1.5 * OMG_A* (1-f_nu))*y[0]]     # [dG/dlna, d^2G/dlna^2]

    def G_ODE_IC_GROWTH(z):
      OMG   = OM_GROWTH
      WG    = W_GROWTH
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3.0*(1.0 + WG)) # H^2(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG))/H2                                  # Omega_DE(z)
      return [1.0, -0.6*(1.0 - WG)*OLG_A]

    def G_ODE_IC_GEO(z):
      OMG   = self.provider.get_param("omegam")
      WG    = self.provider.get_param("w")
      OLG   = (1.0 - OMG)
      H2    = OMG*(1.0 + z)*(1.0 + z)*(1.0 + z) + OLG*(1.0 + z)**(3.0*(1.0 + WG)) # H^2(z)
      OLG_A = OLG*(1.0 + z)**(3.0*(1.0 + WG))/H2                                  # Omega_DE(z)
      return [1.0, -0.6*(1.0 - WG)*OLG_A]

    sol      = odeint(G_GROWTH_ODE, G_ODE_IC_GROWTH(zgs[0]), np.log(1.0/(1.0 + zgs)))[:,0]
    G_growth = np.flip(sol)[0:len(self.z_interp_2D)]
    G_growth = G_growth/G_growth[len(G_growth)-1]

    sol2  = odeint(G_GEO_ODE, G_ODE_IC_GEO(zgs[0]), np.log(1.0/(1.0 + zgs)))[:,0]
    G_geo = np.flip(sol2)[0:len(self.z_interp_2D)]
    G_geo = G_geo/G_geo[len(G_geo)-1]

    G_geo_camb = np.sqrt(PKL.P(self.z_interp_2D, 0.0005)/PKL.P(0, 0.0005))*(1 + self.z_interp_2D)
    G_geo_camb = G_geo_camb/G_geo_camb[len(G_geo)-1] 
    
    G_growth_camb = G_geo_camb * (G_growth/G_geo) # This way minimizes the impact of the small diff 
                                                  # between CAMB growth factor and the ODE above
    lnPL  = np.empty(self.len_pkz_interp_2D)
    lnPNL = np.empty(self.len_pkz_interp_2D)
    t1 = PKNL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    t2 = PKL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    t3 = 2.0*np.log(G_growth_camb/G_geo_camb)
    log10k_interp_2D = self.log10k_interp_2D - np.log10(h) # Cosmolike (and emuls): h/Mpc Units

    for i in range(self.len_z_interp_2D):
      lnPL[i::self.len_z_interp_2D]  = t2[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]
      lnPL[i::self.len_z_interp_2D] += t3[i]
    lnPL += np.log((h**3)) 
  
    if self.non_linear_emul == 1:
      params = {
        'Omm'  : OM_GROWTH,
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : W_GROWTH,
        'wa'   : 0.0
      }

      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      kbt, tmp_bt = self.emulator.get_boost(params, self.z_interp_2D, kbt)
      logkbt = np.log10(kbt)

      for i in range(self.len_z_interp_2D):    
        interp = interp1d(logkbt, np.log(tmp_bt[i]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt
    
    elif self.non_linear_emul == 2:      
      
      for i in range(self.len_z_interp_2D):
        lnPNL[i::self.len_z_interp_2D]  = t1[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]  
        lnPNL[i::self.len_z_interp_2D] += t3[i]
      lnPNL += np.log((h**3))

    elif self.non_linear_emul == 3: #Emulator for w0wa; do NOT support geo-growth split      
      
      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : self.provider.get_param("w"),
        'wa'   : self.provider.get_param("wa")
      }

      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      kbt, tmp_bt = self.emulator.get_boost(params, self.z_interp_2D, kbt)
      logkbt = np.log10(kbt)

      for i in range(self.len_z_interp_2D):    
        interp = interp1d(logkbt, np.log(tmp_bt[i]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt

    elif self.non_linear_emul == 4: #Emulator for w0wa; use Casarini approx; should NOT use this, just use emul=3 with real N-body sim    
      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : -1.0, # w_eff for casarini prescription; to be calculated later in the for loop
        'wa'   : 0.0
      }

      zstar = self.provider.get_param("zstar")
      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))


      for i in range(self.len_z_interp_2D):  
        z_eff = self.z_interp_2D[i]

        ### calculate w_eff as in 1601.07230; Ignored radiation
        def integrad_w_eff(x, w_eff):
          Omm = params['Omm']
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + (1-Omm)*(1.0 + x)**(3.0*(1.0 + w_eff)) # eq 2.1
          return 1/np.sqrt(E2)
        def integrad_w0wa(x):
          Omm = params['Omm']
          w0 = self.provider.get_param("w") 
          wa = self.provider.get_param("wa") 
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + (1-Omm)*(1.0 + x)**(3.0*(1.0 + w0+wa)*np.exp(-3*wa*x/(1+x)) ) # eq 2.1
          return 1/np.sqrt(E2)
        def eq_chi(w_eff):
            return quad(integrad_w0wa, z_eff, zstar)[0] - quad(integrad_w_eff, z_eff, zstar,args=(w_eff))[0]
        w_eff = optimize.root_scalar(eq_chi, bracket=[-3, 3], method='brentq').root

        params['w'] = w_eff #Use wCDM emulator
        kbt, tmp_bt = self.emulator.get_boost(params, z_eff, kbt)
        logkbt = np.log10(kbt)  
        interp = interp1d(logkbt, np.log(tmp_bt[0]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)
        

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt

    ### Emul 5 start
    elif self.non_linear_emul == 5: #Emulator for 4 w-bin    
      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : -1.0, # w_eff for casarini prescription; to be calculated later in the for loop
        'wa'   : 0.0
      }

      zstar = self.provider.get_param("zstar")
      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      OML = 1- self.provider.get_param("omegam")
      w0 = self.provider.get_param("w") 
      w1 = self.provider.get_param("w1") 
      w2 = self.provider.get_param("w2") 
      w3 = self.provider.get_param("w3")
      z1 = 0.7
      z2 = 1.4
      z3 = 2.1
      z4 = 3.0
      fac1 = (1+z1)**(3*(1 + w0))
      fac2 = fac1 * ((1+z2) / (1+z1))**(3*(1 + w1))
      fac3 = fac2 * ((1+z3) / (1+z2))**(3*(1 + w2))
      fac4 = fac3 * ((1+z4) / (1+z3))**(3*(1 + w3))

      Omm = params['Omm']
      Omega_r = 4.18343*10**(-5)/(h**2)

      test = np.zeros(self.len_z_interp_2D)
      for i in range(self.len_z_interp_2D):  
        z_eff = self.z_interp_2D[i]

        ### calculate w_eff as in 1601.07230; Ignored radiation
        def integrad_w_eff(x, w_eff):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + Omega_r*(1.0 + x)**4 + (1-Omm)*(1.0 + x)**(3.0*(1.0 + w_eff)) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)

        def OML_wbin(x):
          a = 1/(1+x)
          if x<=z1:
            return OML*(1+x)**(3*(1+w0))
          elif z1<x<=z2:
            return OML * fac1 * (a * (1+z1))**(-3*(1+w1))
          elif z2<x<=z3:
            return OML * fac2 * (a * (1+z2))**(-3*(1+w2))
          elif z3<x<=z4:
            return OML * fac3 * (a * (1+z3))**(-3*(1+w3))
          elif x>z4:
            return OML * fac4

        def integrad_wbin(x):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + + Omega_r*(1.0 + x)**4 + OML_wbin(x) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)
          
        def eq_chi(w_eff):
            return quad(integrad_wbin, z_eff, zstar)[0] - quad(integrad_w_eff, z_eff, zstar,args=(w_eff))[0]

        w_eff = optimize.root_scalar(eq_chi, bracket=[-3, 3], method='brentq').root

        if w_eff < -1.3:
          print("enforcing EE bound")
          w_eff = -1.3 #Euclid Emulator bound
        elif w_eff > -0.7:
          print("enforcing EE bound")
          w_eff = -0.7

        params['w'] = w_eff #Use wCDM emulator
        kbt, tmp_bt = self.emulator.get_boost(params, z_eff, kbt)
        logkbt = np.log10(kbt)  
        interp = interp1d(logkbt, np.log(tmp_bt[0]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)
        

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt

        test[i] = w_eff

      #np.savetxt("test_wbin.txt", test)
    ### Emul 5 end

    ### Emul 6 Start
    elif self.non_linear_emul == 6: #Emulator for 2 w-bin (same as 4-bin, with different z; check yaml)   
      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : -1.0, # w_eff for casarini prescription; to be calculated later in the for loop
        'wa'   : 0.0
      }

      zstar = self.provider.get_param("zstar")
      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      OML = 1- self.provider.get_param("omegam")
      w0 = self.provider.get_param("w") 
      w1 = self.provider.get_param("w1") 
      w2 = self.provider.get_param("w2") 
      w3 = self.provider.get_param("w3")
      z1 = 0.5
      z2 = 1
      z3 = 1.5
      z4 = 3.0
      fac1 = (1+z1)**(3*(1 + w0))
      fac2 = fac1 * ((1+z2) / (1+z1))**(3*(1 + w1))
      fac3 = fac2 * ((1+z3) / (1+z2))**(3*(1 + w2))
      fac4 = fac3 * ((1+z4) / (1+z3))**(3*(1 + w3))

      Omm = params['Omm']
      Omega_r = 4.18343*10**(-5)/(h**2)

      test = np.zeros(self.len_z_interp_2D)
      for i in range(self.len_z_interp_2D):  
        z_eff = self.z_interp_2D[i]

        ### calculate w_eff as in 1601.07230; Ignored radiation
        def integrad_w_eff(x, w_eff):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + Omega_r*(1.0 + x)**4 + (1-Omm)*(1.0 + x)**(3.0*(1.0 + w_eff)) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)

        def OML_wbin(x):
          a = 1/(1+x)
          if x<=z1:
            return OML*(1+x)**(3*(1+w0))
          elif z1<x<=z2:
            return OML * fac1 * (a * (1+z1))**(-3*(1+w1))
          elif z2<x<=z3:
            return OML * fac2 * (a * (1+z2))**(-3*(1+w2))
          elif z3<x<=z4:
            return OML * fac3 * (a * (1+z3))**(-3*(1+w3))
          elif x>z4:
            return OML * fac4

        def integrad_wbin(x):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + Omega_r*(1.0 + x)**4 + OML_wbin(x) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)
          
        def eq_chi(w_eff):
            return quad(integrad_wbin, z_eff, zstar)[0] - quad(integrad_w_eff, z_eff, zstar,args=(w_eff))[0]

        w_eff = optimize.root_scalar(eq_chi, bracket=[-3, 3], method='brentq').root

        if w_eff < -1.3:
          print("enforcing EE bound")
          w_eff = -1.3 #Euclid Emulator bound
        elif w_eff > -0.7:
          print("enforcing EE bound")
          w_eff = -0.7

        params['w'] = w_eff #Use wCDM emulator
        kbt, tmp_bt = self.emulator.get_boost(params, z_eff, kbt)
        logkbt = np.log10(kbt)  
        interp = interp1d(logkbt, np.log(tmp_bt[0]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)
        

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt

        test[i] = w_eff

      #np.savetxt("test_wbin.txt", test)
    ### Emul 6 end

    ### Emul 7 Start
    elif self.non_linear_emul == 7: #Emulator for 2 w-bin; fix w_eff to that at z=0 (same as 4-bin, with different z; check yaml)   
      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : -1.0, # w_eff for casarini prescription; to be calculated later in the for loop
        'wa'   : 0.0
      }

      zstar = self.provider.get_param("zstar")
      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      OML = 1- self.provider.get_param("omegam")
      w0 = self.provider.get_param("w") 
      w1 = self.provider.get_param("w1") 
      w2 = self.provider.get_param("w2") 
      w3 = self.provider.get_param("w3")
      z1 = 0.5
      z2 = 1
      z3 = 1.5
      z4 = 3.0
      fac1 = (1+z1)**(3*(1 + w0))
      fac2 = fac1 * ((1+z2) / (1+z1))**(3*(1 + w1))
      fac3 = fac2 * ((1+z3) / (1+z2))**(3*(1 + w2))
      fac4 = fac3 * ((1+z4) / (1+z3))**(3*(1 + w3))

      Omm = params['Omm']
      Omega_r = 4.18343*10**(-5)/(h**2)

      test = np.zeros(self.len_z_interp_2D)
      for i in range(self.len_z_interp_2D):  
        z_eff = self.z_interp_2D[0]

        ### calculate w_eff as in 1601.07230; Ignored radiation
        def integrad_w_eff(x, w_eff):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + Omega_r*(1.0 + x)**4 + (1-Omm)*(1.0 + x)**(3.0*(1.0 + w_eff)) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)

        def OML_wbin(x):
          a = 1/(1+x)
          if x<=z1:
            return OML*(1+x)**(3*(1+w0))
          elif z1<x<=z2:
            return OML * fac1 * (a * (1+z1))**(-3*(1+w1))
          elif z2<x<=z3:
            return OML * fac2 * (a * (1+z2))**(-3*(1+w2))
          elif z3<x<=z4:
            return OML * fac3 * (a * (1+z3))**(-3*(1+w3))
          elif x>z4:
            return OML * fac4

        def integrad_wbin(x):
          E2 = Omm*(1.0 + x)*(1.0 + x)*(1.0 + x) + Omega_r*(1.0 + x)**4 + OML_wbin(x) # eq E^2 = H^2/H0^2
          return 1/np.sqrt(E2)
          
        def eq_chi(w_eff):
            return quad(integrad_wbin, z_eff, zstar)[0] - quad(integrad_w_eff, z_eff, zstar,args=(w_eff))[0]

        w_eff = optimize.root_scalar(eq_chi, bracket=[-3, 3], method='brentq').root

        if w_eff < -1.3:
          print("enforcing EE bound")
          w_eff = -1.3 #Euclid Emulator bound
        elif w_eff > -0.7:
          print("enforcing EE bound")
          w_eff = -0.7

        params['w'] = w_eff #Use wCDM emulator
        kbt, tmp_bt = self.emulator.get_boost(params, z_eff, kbt)
        logkbt = np.log10(kbt)  
        interp = interp1d(logkbt, np.log(tmp_bt[0]), kind='linear', 
          fill_value='extrapolate', assume_sorted=True)
        

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

        lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt

        test[i] = w_eff

      #np.savetxt("test_wbin.txt", test)
    ### Emul 7 end

    elif self.non_linear_emul == 100: #Using only linear power spectrum, for determining scale cuts
      lnPNL = lnPL  
    
    else:
      raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)
    
    #Growth-Split (gs) ENDS:

    # Compute chi(z) - convert to Mpc/h
    chi = self.provider.get_comoving_radial_distance(self.z_interp_1D) * h

    cache_alert = self.set_cache_alert(chi, lnPL, lnPNL)

    ci.set_cosmological_parameters(
      omega_matter = self.provider.get_param("omegam"), #Intrinsic Aligment - growth of geometry?
      hubble = self.provider.get_param("H0"),
      is_cached = cache_alert
    )

    if cache_alert == False :
      self.do_cache_chi = np.copy(chi)

      self.do_cache_lnPL = np.copy(lnPL)

      self.do_cache_lnPNL = np.copy(lnPNL)

      self.do_cache_cosmo = np.array([
        self.provider.get_param("omegam"),
        self.provider.get_param("H0")
      ])

      ci.init_linear_power_spectrum(log10k=log10k_interp_2D, z=self.z_interp_2D, lnP=lnPL)

      ci.init_non_linear_power_spectrum(log10k=log10k_interp_2D, z=self.z_interp_2D, lnP=lnPNL)

      #Growth-Split (gs) BEGINS:
      ci.init_growth(z = self.z_interp_2D, G = G_growth_camb)
      #Growth-Split (gs) ENDS:

      ci.init_distances(z = self.z_interp_1D, chi = chi)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_source_related(self, **params_values):
    ci.set_nuisance_shear_calib(
      M = [
        params_values.get(p, None) for p in [
          "LSST_M"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )

    ci.set_nuisance_shear_photoz(
      bias = [
        params_values.get(p, None) for p in [
          "LSST_DZ_S"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )

    ci.set_nuisance_ia(
      A1 = [
        params_values.get(p, None) for p in [
          "LSST_A1_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      A2 = [
        params_values.get(p, None) for p in [
          "LSST_A2_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      B_TA = [
        params_values.get(p, None) for p in [
          "LSST_BTA_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
    )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params_values):
    ci.set_nuisance_bias(
      B1 = [
        params_values.get(p, None) for p in [
          "LSST_B1_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B2 = [
        params_values.get(p, None) for p in [
          "LSST_B2_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B_MAG = [
        params_values.get(p, None) for p in [
          "LSST_BMAG_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_nuisance_clustering_photoz(
      bias = [
        params_values.get(p, None) for p in [
          "LSST_DZ_L"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_point_mass(
      PMV = [
        params_values.get(p, None) for p in [
          "LSST_PM"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )

  # ------------------------------------------------------------------------
  # --------------------------- baryonic PCAs ------------------------------
  # ------------------------------------------------------------------------

  def set_baryon_related(self, **params_values):
    self.baryon_pcs_qs[0] = params_values.get("LSST_BARYON_Q1", 0.0)
    self.baryon_pcs_qs[1] = params_values.get("LSST_BARYON_Q2", 0.0)
    self.baryon_pcs_qs[2] = params_values.get("LSST_BARYON_Q3", 0.0)
    self.baryon_pcs_qs[3] = params_values.get("LSST_BARYON_Q4", 0.0)
    
  def add_baryon_pcs_to_datavector(self, datavector):    
    return datavector[:] + self.baryon_pcs_qs[0]*self.baryon_pcs[:,0] \
      + self.baryon_pcs_qs[1]*self.baryon_pcs[:,1] \
      + self.baryon_pcs_qs[2]*self.baryon_pcs[:,2] \
      + self.baryon_pcs_qs[3]*self.baryon_pcs[:,3]

  def compute_dm_datavector_masked_reduced_dim(self, **params_values):

    self.force_cache_false = True

    ci.init_baryons_contamination(False, "")

    self.set_cosmo_related()

    self.force_cache_false = False

    if (self.probe != "xi"):
      self.set_lens_related(**params_values)

    if (self.probe != "wtheta"):
      self.set_source_related(**params_values)

    # datavector C++ returns a list (not numpy array)
    return np.array(ci.compute_data_vector_masked_reduced_dim())

  # Hack to create baryonic PCAs
  def compute_barion_datavector_masked_reduced_dim(self, sim, **params_values):

    self.force_cache_false = True

    ci.init_baryons_contamination(True, sim)

    self.set_cosmo_related()

    self.force_cache_false = False

    if (self.probe != "xi"):
      self.set_lens_related(**params_values)

    if (self.probe != "wtheta"):
      self.set_source_related(**params_values)

    # datavector C++ returns a list (not numpy array)
    return np.array(ci.compute_data_vector_masked_reduced_dim())

  def generate_baryonic_PCA(self, **params_values):

    cov_L_cholesky = np.linalg.cholesky(
      ci.get_covariance_masked_reduced_dim())

    inv_cov_L_cholesky = np.linalg.inv(cov_L_cholesky)

    ndata_reduced = ci.get_nreduced_dim()

    nbaryons_scenario = ci.get_baryon_pca_nscenarios()

    modelv_dm = self.compute_dm_datavector_masked_reduced_dim(**params_values)

    baryon_diff = np.zeros(shape=(ndata_reduced, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      modelv_baryon = self.compute_barion_datavector_masked_reduced_dim(
        ci.get_baryon_pca_scenario_name(i), **params_values)

      baryon_diff[:,i] = (modelv_baryon-modelv_dm)

    baryon_weighted_diff = np.dot(inv_cov_L_cholesky, baryon_diff)

    U, Sdig, VT = np.linalg.svd(baryon_weighted_diff, full_matrices=True)

    # MAKE SURE WHATEVER VERSION OF NP HAVE U IN THE RIGHT ORDER
    if(np.all(np.diff(Sdig) <= 0) != True):
      raise LoggedError(self.log, "LOGICAL ERROR WITH NUMPY FUNCTION GEN PCA")

    PCs = np.empty(shape=(ndata_reduced, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      PCs[:,i] = U[:,i]

    PCs = np.dot(cov_L_cholesky, PCs)

    # Now we need to expand the number of dimensions
    ndata = ci.get_ndim()
    PCS_FINAL = np.empty(shape=(ndata, nbaryons_scenario))

    for i in range(nbaryons_scenario):
      PCS_FINAL[:,i] = ci.get_expand_dim_from_masked_reduced_dim(PCs[:,i])

    np.savetxt(self.filename_baryon_pca, PCS_FINAL)

    ci.init_baryons_contamination(False,"")
