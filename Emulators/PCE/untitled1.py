params = {
  'Omm'  : self.provider.get_param("omegam"),
  'As'   : self.provider.get_param("As"),
  'Omb'  : self.provider.get_param("omegab"),
  'ns'   : self.provider.get_param("ns"),
  'h'    : self.provider.get_param("H0")/100.0,
  'mnu'  : self.provider.get_param("mnu"), 
  'w'    : -1,
  'wa'   : 0.0
}
print(params['mnu'])  
#These kbt are in units of h/Mpc .
kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D)) # Need to return these ks in emulator
kbt, tmp_bt = self.emulator.get_boost(params, self.z_interp_2D, kbt)
logkbt = np.log10(kbt)
   
interp = interp1d(logkbt, 
  np.log(tmp_bt[i]), 
  kind = 'linear', 
  fill_value = 'extrapolate', 
  assume_sorted = True
)
lnbt = interp(log10k_interp_2D)
lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0

lnPNL[i::self.len_z_interp_2D] = lnPL[i::self.len_z_interp_2D] + lnbt 