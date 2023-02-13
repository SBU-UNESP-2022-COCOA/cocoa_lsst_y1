import numpy as np
import scipy
import scipy.stats
import scipy.interpolate
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import CubicSpline
from scipy.fftpack import dst, idst
import scipy.integrate
import math
import sys, platform, os
import GPy

def get_redshifts(z_ini, output_redshifts, timestep_nsteps):
    redshift_endpoints = [z_ini] + output_redshifts
    scale_endpoints = [1/(1+z) for z in redshift_endpoints]
    das = [(scale_endpoints[i+1] - scale_endpoints[i])/timestep_nsteps[i] for i in range(len(timestep_nsteps))]
    scales = []
    for i in range(len(timestep_nsteps)):
        for j in range(timestep_nsteps[i]):
            scale = scale_endpoints[i] + j*das[i]
            scales.append(scale)
    zs_ = [round((1/a - 1.0), 3) for a in scales]
    zs_.append(0.0)
    zs_= np.flip(zs_)
    return zs_

param_dim = 5
k_min = 1e-2
k_max = 3
n_k_bins = 200
num_points = 400
num_points_test = 10
redshifts = get_redshifts(20, [3,2,1,0.5,0], [12,5,8,9,17])
redshifts_ee2 = [redshifts[i] for i in range(len(redshifts)-3)]
N_pc = 6
param_mins = [0.24, 0.04, 0.92, 1.7e-9, 0.61]
param_maxs = [0.4, 0.06, 1.0, 2.5e-9, 0.73]

def find_first_minimum(array):
    for i, entry in enumerate(array):
        if i <= 10:
            continue
        else:
            left_neighbor = array[i-1]
            right_neighbor = array[i+1]
            if entry < left_neighbor and entry < right_neighbor:
                return i, entry
        if i == len(array)-1:
            return 'Error'
def find_second_maximum(array):
    maxima = 0
    for i, entry in enumerate(array):
        if i <= 10:
            continue
        else:
            left_neighbor = array[i-1]
            right_neighbor = array[i+1]
            if entry > left_neighbor and entry > right_neighbor:
                maxima += 1
                if maxima == 2:
                    return i, entry
        if i == len(array) - 1:
            return 'Error'
def smooth_bao(ks, pk):
    spline_loglog_pk = CubicSpline(np.log(ks), np.log(pk))
    n = 10
    dst_ks = np.linspace(ks[0], ks[-1], 2**n)
    logks = np.log(dst_ks)
    logkpk = logks + spline_loglog_pk(logks)
    sine_transf_logkpk = dst(logkpk, type=2, norm='ortho')
    odds = [] # odd entries
    evens = [] # even entries
    even_is = [] # odd indices
    odd_is = [] # even indices
    all_is = [] # all indices
    for i, entry in enumerate(sine_transf_logkpk):
        all_is.append(i)
        if i%2 == 0:
            even_is.append(i)
            evens.append(entry)
        else:
            odd_is.append(i)
            odds.append(entry)
    odd_is = np.array(odd_is)
    even_is = np.array(even_is)
    odds_interp = CubicSpline(odd_is, odds)
    evens_interp = CubicSpline(even_is, evens)
    d2_odds = (odds_interp.derivative(nu=2))
    d2_evens = (evens_interp.derivative(nu=2))
    d2_odds_avg = (d2_odds(odd_is) + d2_odds(odd_is + 2) + d2_odds(odd_is - 2))/3
    d2_evens_avg = (d2_evens(even_is) + d2_evens(even_is + 2) + d2_evens(even_is - 2))/3
    i_star_bottom, _ = find_first_minimum(d2_odds_avg)
    i_star_top, _ = find_second_maximum(d2_odds_avg)
    imin_odd = i_star_bottom - 3
    imax_odd = i_star_top + 20
    i_star_bottom, _ = find_first_minimum(d2_evens_avg)
    i_star_top, _ = find_second_maximum(d2_evens_avg)
    imin_even = i_star_bottom - 3
    imax_even = i_star_top + 10   
    odd_is_removed_bumps = []
    odds_removed_bumps = []
    for i, entry in enumerate(odds):
        if i in range(imin_odd, imax_odd+1):
            continue
        else:
            odd_is_removed_bumps.append(2*i+1)
            odds_removed_bumps.append(entry)
    even_is_removed_bumps = []
    evens_removed_bumps = []
    for i, entry in enumerate(evens):
        if i in range(imin_even, imax_even+1):
            continue
        else:
            even_is_removed_bumps.append(2*i)
            evens_removed_bumps.append(entry)
    odd_is_removed_bumps = np.array(odd_is_removed_bumps)
    even_is_removed_bumps = np.array(even_is_removed_bumps)
    odds_removed_spline_iplus1 = CubicSpline(odd_is_removed_bumps, (odd_is_removed_bumps+1)**2 * odds_removed_bumps)
    evens_removed_spline_iplus1 = CubicSpline(even_is_removed_bumps, (even_is_removed_bumps+1)**2 * evens_removed_bumps)
    odds_treated_iplus1 = odds_removed_spline_iplus1(odd_is)
    evens_treated_iplus1 = evens_removed_spline_iplus1(even_is)
    odds_treated = odds_treated_iplus1/(odd_is+1)**2
    evens_treated = evens_treated_iplus1/(even_is+1)**2
    treated_transform = []
    for odd, even in zip(odds_treated, evens_treated):
        treated_transform.append(even)
        treated_transform.append(odd)
    treated_logkpk = idst(treated_transform, type=2, norm='ortho')
    pk_nw = np.exp(treated_logkpk)/dst_ks
    pk_nw_spline = CubicSpline(dst_ks, pk_nw)
    pk_nw = pk_nw_spline(ks)
    return pk_nw
def smear_bao(ks, pk, pk_nw):
    integral = scipy.integrate.simps(pk,ks)
    k_star_inv = (1.0/(3.0 * math.pi**2)) * integral
    Gk = np.array([np.exp(-0.5*k_star_inv * (k_**2)) for k_ in ks])
    pk_smeared = pk*Gk + pk_nw*(1.0 - Gk)
    return pk_smeared
# def PC_project(data, n_pc):
#     pca = PCA(n_components = n_pc)
#     pca.fit(data)
#     transformed_data = pca.fit_transform(data)
#     return transformed_data, pca
# def turn_qk_to_b(params_, z_index, qk):
#     index = lhs_test.index(params_)
#     pk_l = [pks_l_test[z_index][index]]
#     pk_nw = smooth_bao(ks, pk_l[0])
#     pk_smeared = smear_bao(ks, pk_l[0], pk_nw)
#     b = (pk_smeared * np.exp(qk))/pk_l[0]
#     return b
def find_worst_points(uncertainties, frac_to_keep, lhs_iter):
    num_to_keep = int(frac_to_keep*len(uncertainties))
    maxs = []
    samples = []
    for i in range(len(uncertainties)):
        maxs.append(np.max(uncertainties[i]))
    sorted_maxs = np.flip(np.sort(maxs))
    for i in range(num_to_keep):
        if len(samples) >= num_to_keep:
            break
        for j in range(len(maxs)):
            if sorted_maxs[i] == maxs[j] :
                samples.append(j)
    lhs_kept = [lhs_iter[sample] for sample in samples]
    return lhs_kept
def normalize_param(param_min, param_max, param):
    normalized_param = (param - param_min)/(param_max - param_min)
    #normalized_param = param
    return normalized_param
def unnormalize_param(param_min, param_max, normalized_param):
    param = (normalized_param * (param_max - param_min)) + param_min
    #param = normalized_param
    return param
# def get_qs_and_pcas(pks_l, pks_nl, N_pc, n_iter):
#     qs_full = []
#     qs_reduced = []
#     pcas = []
#     for j in range(len(redshifts_ee2)):
#         qs_full.append([])
#         qs_reduced.append([])
#         pcas.append([])
#         for i in range(num_points + int(n_iter*frac_to_keep*lhs_iter_size)):
#             pk_nw = smooth_bao(ks, pks_l[j][i])
#             pk_smeared = smear_bao(ks, pks_l[j][i], pk_nw)
#             q_bacco = np.log(pks_nl[j][i]/pk_smeared)
#             qs_full[j].append(q_bacco)
#         qs_reduced[j], pcas[j] = PC_project(qs_full[j], N_pc)
#     return pcas, qs_reduced, qs_full
def initialize_emulator(qs_reduced_,lhs):
    all_gps = []
    for z_index in range(len(redshifts_ee2)):
        all_gps.append([])
        for pc_index in range(N_pc):
            x = []
            y=[]
            for i in range(len(qs_reduced_[z_index])):
                x.append(lhs[i])
                q_of_k = qs_reduced_[z_index][i][pc_index]
                y.append([q_of_k])
            kernel = GPy.kern.RBF(input_dim=param_dim, ARD = True)
            x=np.array(x)
            y=np.array(y)
            m = GPy.models.GPRegression(x,y,kernel)
            m.optimize()
            all_gps[z_index].append(m)
    return all_gps
def inv_pc(pcs_,mean_,vec):
    expanded_q = [mean_[i] for i in range(len(mean_))]
    for pc_index in range(len(pcs_)):
        expanded_q += vec[pc_index]*pcs_[pc_index]
    return expanded_q
def emulate_all_zs(params_, all_gps, qs_reduced, all_pcs_, all_means_, ks_in, ks_out, zs_in, zs_out):
    emulated_qs_ = []
    emulation_uncertainties = []
    for z_index in range(len(redshifts_ee2)):
        emulated_reduced_q = []
        emulation_uncertainty2 = 0
        for pc_index in range(N_pc):
            params_to_predict = np.array([params_])
            m = all_gps[z_index][pc_index]
            pred, pred_var = m.predict(params_to_predict)
            emulated_reduced_q.append(pred[0][0])
            emulation_uncertainty2 = emulation_uncertainty2 + pred_var[0]**2
            the_mean = [entry for entry in all_means_[z_index]]
        emulated_q = inv_pc(all_pcs_[z_index], the_mean,emulated_reduced_q)
        #emulated_b = turn_qk_to_b(params_, z_index, emulated_q)
        emulated_qs_.append(emulated_q)
        emulation_uncertainty = math.sqrt(emulation_uncertainty2)
        emulation_uncertainties.append(emulation_uncertainty)
    emulated_qs_interp = RectBivariateSpline(zs_in, ks_in, emulated_qs_)
    emulated_qs = emulated_qs_interp(zs_out, ks_out)
    return emulated_qs, emulation_uncertainties
# def reduce_kbins(ks_in, ks_out, cola_vec_full):
#     interpolated_vec = CubicSpline(ks_in, cola_vec_full)
#     cola_vec_reduced = interpolated_vec(ks_out)
#     return cola_vec_reduced
# def increase_kbins(ks_in, ks_out, vec_short):
#     interpolated_vec = CubicSpline(ks_in, vec_short)
#     vec_long = interpolated_vec(ks_out)
#     return vec_long


