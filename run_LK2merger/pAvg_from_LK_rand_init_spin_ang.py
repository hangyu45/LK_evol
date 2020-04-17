#!/usr/bin/env python

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import h5py as h5
import os, sys, argparse

from numba import jit, prange
import timeit

plt.rc('figure', figsize=(9, 7))
plt.rcParams.update({'text.usetex': False,
                     'font.family': 'serif',
                     'font.serif': ['Georgia'],
                     'mathtext.fontset': 'cm',
                     'lines.linewidth': 2.5,
                     'font.size': 20,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.5,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 17,
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.05,
                     'savefig.dpi': 80,
                     'pdf.compression': 9})

sys.path.append('/home/hang.yu/astro/LK_evol/')
from myConstants import *
import LKlib as LK

#######################################################################
### input 
#######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', type=np.int, default=0)
parser.add_argument('--M3', type=np.float, default=1.e9, 
                   help='M3 in [Msun]')
parser.add_argument('--ao', type=np.float, default=0.06, 
                   help='Semi-major axis of the outer orbit in [pc]')
parser.add_argument('--ai-0', type=np.float, default=3., 
                   help='Initial semi-major axis of the inner orbit in [AU]')
parser.add_argument('--plot-flag', type=np.int, default=0)

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)

# run_id=kwargs['run_id']
run_id = kwargs['run_id']
M3, ao, ai_0=kwargs['M3'], kwargs['ao'], kwargs['ai_0']
br_flag, ss_flag=1, 1
plot_flag = kwargs['plot_flag']

fig_dir = '/home/hang.yu/public_html/astro/LK_evol/LK2merger/rand_init_spin_ang/DA/M3_%.1eMs_ao_%.3fpc_ai0_%.1fAU/'\
            %(M3, ao, ai_0)
data_dir = 'data/rand_init_spin_ang/DA/M3_%.1eMs_ao_%.3fpc_ai0_%.1fAU/'\
            %(M3, ao, ai_0)
# prefix = 'id_%i_'%(run_id)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
M3*=Ms
ao*=pc
ai_0*=AU

### read in the data from the end of LK as the initial condition ### 

if os.path.exists(data_dir + 'id_%i_LK_cond.txt'%run_id):
        data_LK = np.loadtxt(data_dir + 'id_%i_LK_cond.txt'%run_id)
        
        prefix = 'id_%i_'%run_id # make sure the prefix is consistent 
        print('File:', run_id)
        
        fid_600 = open(data_dir + prefix + 'r_600_cond.txt', 'w')
        fid_300 = open(data_dir + prefix + 'r_300_cond.txt', 'w')
        fid_100 = open(data_dir + prefix + 'r_100_cond.txt', 'w')
        
        nSamp = data_LK.shape[0]
        for j in range(nSamp):
            M1, M2, chi1, chi2, chi_eff = data_LK[j, :5]
            M1 = M1*Ms
            M2 = M2*Ms
            
            qq = M2/M1
            Mt = M1 + M2
            mu = M1 * M2 / Mt
            eta = mu/Mt
            
            S1 = chi1 * G*M1**2./c
            S2 = chi2 * G*M2**2./c
            
            r_Mt = G*Mt/c**2.
            S_Mt = G*Mt**2./c
            
            a_LK = data_LK[j, 5]*r_Mt
            J_LK, L_LK, S_LK \
                = data_LK[j, 9]*S_Mt, data_LK[j, 10]*S_Mt, data_LK[j, 11]*S_Mt
            
            eff_LK = L_LK/(mu*np.sqrt(G*Mt*a_LK))
            e_LK = np.sqrt(1. - eff_LK**2.)
            
            t_GW_LK = LK.get_inst_t_gw_from_a_orb(M1, M2, a_LK, e_LK) 
#             print('1-e: %e, chi_eff: %f'%(1.-e_LK, chi_eff))
            
            
            ### evolve the scalar a, L, e ###
            par_aL = np.array([M1, M2])

            int_func=lambda loga_, logL_:\
                LK.evol_logL_vs_loga(loga_, logL_, par_aL)
    
            sol=integ.solve_ivp(int_func, \
                t_span=(np.log(a_LK/r_Mt), np.log(6.)), y0=np.array([np.log(L_LK/S_Mt)]), rtol=3e-14, atol=1e-17)

            a_scal = np.exp(sol.t)*r_Mt
            L_scal = np.exp(sol.y[0, :])*S_Mt
            eff_scal = L_scal/(mu*np.sqrt(G*Mt*a_scal))
            e_scal = np.sqrt(1. - eff_scal**2.)

            loge_vs_logL_tck = interp.splrep(np.log(L_scal[::-1]), np.log(e_scal[::-1]))
            logL_vs_loga_tck = interp.splrep(np.log(a_scal[::-1]), np.log(L_scal[::-1]))
            e_vs_L_func=lambda LL: np.exp(interp.splev(np.log(LL), loge_vs_logL_tck))
            L_vs_a_func=lambda aa: np.exp(interp.splev(np.log(aa), logL_vs_loga_tck))
            
            
            ### J vs L function ###
            nPt = 50
            par_JL = np.array([M1, M2, S1, S2, chi_eff])
            int_func=lambda L_nat_, J_nat_:\
                LK.evol_J_avg(L_nat_, J_nat_, e_vs_L_func, par_JL, nPt=nPt)
            
            
            ### evol to 600 r_Mt ###
            L_600 = L_vs_a_func(600.*r_Mt)
            e_600 = e_vs_L_func(L_600)
            sol=integ.solve_ivp(int_func, \
                t_span=(L_LK/S_Mt, L_600/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-14, atol=1e-14)

            J_600 = sol.y[0,-1] * S_Mt
            Sm_600, Sp_600=LK.find_Smp(J_600, L_600, e_600, par_JL)
            
            fid_600.write('%.6f\t%.6f\t%.6f\t%.6f\t%.9f\t%.9e\t%.9e\t%.9e\t%.6e\t%.6e\n'\
                  %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
                    J_600/S_Mt, L_600/S_Mt, e_600, \
                    Sm_600/S_Mt, Sp_600/S_Mt))
            
            
            ### evol to 300 r_Mt ###
            L_300 = L_vs_a_func(300.*r_Mt)
            e_300 = e_vs_L_func(L_300)
            sol=integ.solve_ivp(int_func, \
                t_span=(L_LK/S_Mt, L_300/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-14, atol=1e-14)

            J_300 = sol.y[0,-1] * S_Mt
            Sm_300, Sp_300=LK.find_Smp(J_300, L_300, e_300, par_JL)
            
            fid_300.write('%.6f\t%.6f\t%.6f\t%.6f\t%.9f\t%.9e\t%.9e\t%.9e\t%.6e\t%.6e\n'\
                  %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
                    J_300/S_Mt, L_300/S_Mt, e_300, \
                    Sm_300/S_Mt, Sp_300/S_Mt))
            
            
            ### evol to 100 r_Mt ###
            L_100 = L_vs_a_func(100.*r_Mt)
            e_100 = e_vs_L_func(L_100)
            sol=integ.solve_ivp(int_func, \
                t_span=(L_LK/S_Mt, L_100/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-14, atol=1e-14)

            J_100 = sol.y[0,-1] * S_Mt
            
            print('J_100:', J_100/S_Mt)
            Sm_100, Sp_100=LK.find_Smp(J_100, L_100, e_100, par_JL)
            
            fid_100.write('%.6f\t%.6f\t%.6f\t%.6f\t%.9f\t%.9e\t%.9e\t%.9e\t%.6e\t%.6e\n'\
                  %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
                    J_100/S_Mt, L_100/S_Mt, e_100, \
                    Sm_100/S_Mt, Sp_100/S_Mt))
            
            
        fid_600.close()
        fid_300.close()
        fid_100.close()