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
parser.add_argument('--run-id', type=np.int, default=999)
parser.add_argument('--M3', type=np.float, default=1.e9, 
                   help='M3 in [Msun]')
parser.add_argument('--ao', type=np.float, default=0.06, 
                   help='Semi-major axis of the outer orbit in [pc]')
parser.add_argument('--ai-0', type=np.float, default=3., 
                   help='Initial semi-major axis of the inner orbit in [AU]')
parser.add_argument('--atol', type=np.float, default=3e-10)
parser.add_argument('--rtol', type=np.float, default=3e-10)
parser.add_argument('--plot-flag', type=np.int, default=1)

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)

run_id=kwargs['run_id']
M3, ao, ai_0=kwargs['M3'], kwargs['ao'], kwargs['ai_0']
atol, rtol = kwargs['atol'], kwargs['rtol']
br_flag, ss_flag=1, 1
plot_flag = kwargs['plot_flag']

fig_dir = '/home/hang.yu/public_html/astro/LK_evol/LK2merger/fix_init_spin_ang/DA/'
data_dir = 'data/fix_init_spin_ang/DA/'
prefix = 'test_id_%i_'%(run_id)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
M3*=Ms
ao*=pc
ai_0*=AU

#######################################################################
### MCMC
#######################################################################

## FIXME ##
# fix M1, M2, chi1, chi2
M1, M2 = 85.*Ms, 65.*Ms
chi1, chi2=0.7, 0.7
Tgw_trgt = 1.e9 * P_yr

# only randomize angle
I_io_m, I_io_p = LK.find_merger_window(Tgw_trgt, \
                      M1, M2, M3, ai_0, ao, eo=0.)
print(I_io_m*180./np.pi, I_io_p*180./np.pi)
# I_io_0 = stats.uniform(loc=I_io_m, scale=(I_io_p-I_io_m)).rvs() 
I_io_0 = 88.7*np.pi/180.

# fix the initial spin alignment wrt the total ang momentum
I_S1_0 = 70.*np.pi/180. #np.pi/2.
I_S2_0 = 120.*np.pi/180.  #np.pi/2.
phi_S1_0 = stats.uniform(scale=2.*np.pi).rvs()
phi_S2_0 = stats.uniform(scale=2.*np.pi).rvs()

# I_io_0 = 92.52 * np.pi/180.
# I_S1_0 = 92.52 * np.pi/180.
# phi_S1_0 = 45.*np.pi/180.
# I_S2_0 = 92.52 * np.pi/180.
# phi_S2_0 = 0.*45.*np.pi/180.

uLi_0 = np.array([np.sin(I_io_0), 0., np.cos(I_io_0)])
uS1_0 = np.array([np.sin(I_S1_0)*np.cos(phi_S1_0), 
                  np.sin(I_S1_0)*np.sin(phi_S1_0), 
                  np.cos(I_S1_0)])
uS2_0 = np.array([np.sin(I_S2_0)*np.cos(phi_S2_0), 
                  np.sin(I_S2_0)*np.sin(phi_S2_0),               
                  np.cos(I_S2_0)])

uS1_d_uLi_0 = LK.inner(uS1_0, uLi_0)
uS2_d_uLi_0 = LK.inner(uS2_0, uLi_0)
theta1_SL_0 = np.arccos(uS1_d_uLi_0)
theta2_SL_0 = np.arccos(uS2_d_uLi_0)
print('initial sl alignment:', theta1_SL_0*180./np.pi, theta2_SL_0*180./np.pi)

chi_eff_0 = (M1*uS1_d_uLi_0*chi1 + M2*uS2_d_uLi_0*chi2)/(M1+M2)
print('Initial chi eff', chi_eff_0)

#######################################################################
### Initial vectors
#######################################################################

mu_i = M1*M2/(M1+M2)
mu_o = (M1+M2)*M3/(M1+M2+M3)

Mt = M1+M2
qq = M2/M1
eta = mu_i/Mt

S1 = chi1 * G*M1**2./c
S2 = chi2 * G*M2**2./c

r_Mt=G*(M1+M2)/c**2.
S_Mt=G*(M1+M2)**2./c

ei_0 = 1.e-3
eff_i_0 = np.sqrt(1.-ei_0**2.)
Li_0 = mu_i * np.sqrt(G*(M1+M2)*ai_0)*eff_i_0
eo = 0.
eff_o = np.sqrt(1.-eo**2.)
Lo = mu_o * np.sqrt(G*(M1+M2+M3)*ao)*eff_o

t_GW_0 = LK.get_inst_t_gw_from_a_orb(M1, M2, ai_0, ei_0)
t_LK_0 = LK.get_t_lk(M1, M2, M3, ai_0, ao)
print('T_gw/yr, T_lk/yr:', '%e, %e'%(t_GW_0/P_yr, t_LK_0/P_yr))

e_max = LK.find_ei_max_vs_Ii_0(I_io_0, \
        M1, M2, M3, ai_0, ao, eo=0)
t_GW_exp = LK.get_inst_t_gw_from_a_orb(M1, M2, ai_0, 0)*(1.-e_max**2.)**3.
print('(1-e_max), T_gw_exp/yr', '%e, %e'%(1-e_max, t_GW_exp/P_yr))

t_unit=t_LK_0
Li_unit, Lo_unit = Li_0, Lo
ai_unit = ai_0
S1_unit = S1
S2_unit = S2
par_LK = np.array([M1, M2, M3, ao, \
        t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
        br_flag, ss_flag])

ai_nat = ai_0/ai_unit
Li_nat_v = uLi_0 * Li_0/Li_unit
ei_v = np.array([0., ei_0, 0])
Lo_nat_v = np.array([0., 0., 1.])*Lo/Lo_unit
eo_v = np.zeros(3)
S1_nat_v = uS1_0 * S1/S1_unit
S2_nat_v = uS2_0 * S2/S2_unit

y_nat_init = np.hstack([
    ai_nat, \
    Li_nat_v, ei_v, \
    Lo_nat_v, eo_v, \
    S1_nat_v, S2_nat_v])

#######################################################################
### solve LK ai = ai_0/10
#######################################################################
t_run0 = timeit.default_timer()
@jit(nopython=True)
def terminator(t_nat, y_nat_vect, par):
    # parse parameters
    ai_nat,\
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    M1, M2, M3, ao, \
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
        
    ai = ai_nat * ai_unit
    resi = (ai-ai_0/10.)/ai_unit
    return resi

term_func=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par_LK)
term_func.direction = -1
term_func.terminal = True

int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_LK_quad_da(t_nat_, y_nat_vect_, par_LK)
    
sol=integ.solve_ivp(int_func, \
        t_span=(0, 3e6), y0=y_nat_init, rtol=rtol, atol=atol, \
        events=term_func)

t_run1 = timeit.default_timer()
print('LK run time:', t_run1 - t_run0)

#######################################################################
### LK outputs
#######################################################################

tt = sol.t*t_unit

ai = sol.y[0, :]*ai_unit 

Li_x = sol.y[1, :]*Li_unit 
Li_y = sol.y[2, :]*Li_unit 
Li_z = sol.y[3, :]*Li_unit 
Li = np.sqrt(Li_x**2. + Li_y**2. + Li_z**2.)
ei_x = sol.y[4, :]
ei_y = sol.y[5, :]
ei_z = sol.y[6, :]
ei=np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)

S1_x = sol.y[13, :]*S1_unit
S1_y = sol.y[14, :]*S1_unit
S1_z = sol.y[15, :]*S1_unit

S2_x = sol.y[16, :]*S2_unit 
S2_y = sol.y[17, :]*S2_unit
S2_z = sol.y[18, :]*S2_unit

Jx = Li_x + S1_x + S2_x
Jy = Li_y + S1_y + S2_y
Jz = Li_z + S1_z + S2_z
J = np.sqrt(Jx**2. + Jy**2. + Jz**2.)

S_x = S1_x + S2_x
S_y = S1_y + S2_y 
S_z = S1_z + S2_z
S = np.sqrt(S_x**2. + S_y**2. + S_z**2.)

uS1_d_uLi = (S1_x*Li_x + S1_y*Li_y + S1_z*Li_z)/(S1 * Li)
uS2_d_uLi = (S2_x*Li_x + S2_y*Li_y + S2_z*Li_z)/(S2 * Li)
uS1_d_uS2 = (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)

chi_eff = np.median((M1*uS1_d_uLi[-5:]*chi1 + M2*uS2_d_uLi[-5:]*chi2)/(M1+M2))

theta1_SL = np.arccos(uS1_d_uLi)
theta2_SL = np.arccos(uS2_d_uLi)
theta_SS = np.arccos(uS1_d_uS2)

if plot_flag>0:
    fig=plt.figure(figsize=(9, 14))
    ax=fig.add_subplot(311)
    ax.semilogy(tt/1.e3/P_yr, ai/AU, alpha=0.8, color='tab:grey')
    ax.set_ylim([2.2e-1, 3.8e0])
    ax.set_ylabel(r'$a_{\rm i}$ [AU]')
    ax.set_xticklabels([])
    ax.set_yticks([1])
    ax.set_yticklabels(['1'])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0, 3.0], minor=True)
    ax.set_yticklabels(['0.3', '', '', '', '', '', '', '', '3'], minor=True)

    ax=fig.add_subplot(312)
    ax.semilogy(tt/1.e3/P_yr, 1.-ei, alpha=0.8, color='tab:grey')
    ax.set_ylabel(r'$1-e_{\rm i}$')
    ax.set_xticklabels([])

    ax=fig.add_subplot(313)
    ax.plot(tt/1.e3/P_yr, theta1_SL*180./np.pi, alpha=0.8, label=r'$\theta_{S_1L}$', color='tab:grey')
    ax.plot(tt/1.e3/P_yr, theta2_SL*180./np.pi, alpha=0.8, label=r'$\theta_{S_2L}$', color='tab:olive')
#     ax.axhline(90., color='tab:red', ls=':', alpha=0.5)
    ax.axhline(I_S1_0*180./np.pi, color='tab:red', ls=':', label=r'$\theta_{S_1L_{\rm o}}^{(0)}$')
    ax.axhline(I_S2_0*180./np.pi, color='tab:purple', ls=':', label=r'$\theta_{S_2L_{\rm o}}^{(0)}$')

    ax.set_ylabel(r'$\theta_{SL}$ [$^\circ$]')
    ax.legend(loc='upper left', framealpha=0.85)
    ax.set_xlabel(r'Time [kyr]')

#     ax=fig.add_subplot(414)
#     ax.plot(tt/1.e9/P_yr, theta_SS*180./np.pi, alpha=0.8, color='tab:grey')
#     ax.set_ylabel(r'$\theta_{SS}$ [$^\circ$]')
#     ax.set_xlabel(r'Time [Gyr]')

    fig.subplots_adjust(hspace=0)
    fig.savefig(fig_dir+prefix+'LK_evol.pdf')
    plt.close()
    
print('a_i (end of LK) / ai_0', ai[-1]/ai_0)
if ai[-1]/ai_0>0.11:
    # LK failed converging
    sys.exit()

# proceed & record data only if LK converged


# #######################################################################
# ### evolve (scalars) a, L, e to merger
# #######################################################################

# ## quantities at the end of the LK run
# a_LK=ai[-1]
# L_LK=Li[-1]
# J_LK=J[-1]

# # e_LK=ei[-1]
# eff_LK = L_LK/(mu_i*np.sqrt(G*(M1+M2)*a_LK))
# e_LK = np.sqrt(1. - eff_LK**2.)

# t_GW_LK = LK.get_inst_t_gw_from_a_orb(M1, M2, a_LK, e_LK) 
# print('t_GW when a=a0/10: [yr]', t_GW_LK/P_yr)
# print('a/AU, e, e/e_ref', a_LK/AU, e_LK, e_LK/ei[-1])
# print('chi_eff', chi_eff)
# print('L/Mt^2, J/Mt^2', L_LK/S_Mt, J_LK/S_Mt)

# par_aL = np.array([M1, M2])

# int_func=lambda loga_, logL_:\
#     LK.evol_logL_vs_loga(loga_, logL_, par_aL)
    
# sol=integ.solve_ivp(int_func, \
#         t_span=(np.log(a_LK/r_Mt), np.log(6.)), y0=np.array([np.log(L_LK/S_Mt)]), rtol=3e-14, atol=1e-17)

# a_scal = np.exp(sol.t)*r_Mt
# L_scal = np.exp(sol.y[0, :])*S_Mt
# eff_scal = L_scal/(mu_i*np.sqrt(G*(M1+M2)*a_scal))
# e_scal = np.sqrt(1. - eff_scal**2.)

# print(L_scal[-1]/S_Mt)

# loge_vs_logL_tck = interp.splrep(np.log(L_scal[::-1]), np.log(e_scal[::-1]))
# logL_vs_loga_tck = interp.splrep(np.log(a_scal[::-1]), np.log(L_scal[::-1]))
# e_vs_L_func=lambda LL: np.exp(interp.splev(np.log(LL), loge_vs_logL_tck))
# L_vs_a_func=lambda aa: np.exp(interp.splev(np.log(aa), logL_vs_loga_tck))

# #######################################################################
# ### J vs L func
# #######################################################################

# nPt = 50
# par_JL = np.array([M1, M2, S1, S2, chi_eff])
# int_func=lambda L_nat_, J_nat_:\
#     LK.evol_J_avg(L_nat_, J_nat_, e_vs_L_func, par_JL, nPt=nPt)

# #######################################################################
# ### evol to 600 r_Mt
# #######################################################################

# t_run0 = timeit.default_timer()

# L_600 = L_vs_a_func(600.*r_Mt)
# e_600 = e_vs_L_func(L_600)
# sol=integ.solve_ivp(int_func, \
#     t_span=(L_LK/S_Mt, L_600/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-12, atol=1e-12)

# J_600 = sol.y[0,-1] * S_Mt
# print('r = 600 M')
# print('J, L, e', J_600/S_Mt, L_600/S_Mt, e_600)

# Sm_600, Sp_600=LK.find_Smp(J_600, L_600, e_600, par_JL)
# t_run1 = timeit.default_timer()
# print('dJdL run (to 600 M):', t_run1 - t_run0)

#######################################################################
### record data
#######################################################################

# ## record initial condition
# fid = open(data_dir + prefix + 'init_cond.txt', 'a')
# fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'\
#           %(M1/Ms, M2/Ms, chi1, chi2, chi_eff_0, \
#             I_io_0, \
#             I_S1_0, phi_S1_0, \
#             I_S2_0, phi_S2_0, \
#             theta1_SL_0, theta2_SL_0, \
#             0.))
# fid.close()

# ## record condition at the end of LK
# fid = open(data_dir + prefix + 'LK_cond.txt', 'a')
# fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.9f\t%.6e\t%.9e\t%.9e\t%.6e\t%.9e\t%.9e\t%.9e\t%.6f\t%.6f\t%.6f\n'\
#           %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
#             ai[-1]/r_Mt, 1-ei[-1], 1-np.max(ei), tt[-1]/P_yr,\
#             J[-1]/S_Mt, Li[-1]/S_Mt, S[-1]/S_Mt, \
#             theta1_SL[-1], theta2_SL[-1], theta_SS[-1]))
# fid.close()

# fid = open(data_dir + prefix + 'r_600_cond.txt', 'a')
# fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.9f\t%.9e\t%.9e\t%.9e\t%.6e\t%.6e\n'\
#           %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
#             J_600/S_Mt, L_600/S_Mt, e_600, \
#             Sm_600/S_Mt, Sp_600/S_Mt))
# fid.close()

#######################################################################
### evol to ISCO
#######################################################################

# t_run0 = timeit.default_timer()

# L_isco = L_vs_a_func(6.*r_Mt)
# e_isco = e_vs_L_func(L_isco)
# sol=integ.solve_ivp(int_func, \
#     t_span=(L_LK/S_Mt, L_isco/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-14, atol=1e-17)

# J_isco = sol.y[0,-1] * S_Mt
# print('r = 6 M')
# print('J, L, e', J_isco/S_Mt, L_isco/S_Mt, e_isco)

# Sm_isco, Sp_isco=LK.find_Smp(J_isco, L_isco, e_isco, par_JL)

# fid = open(data_dir + prefix + 'r_isco_cond.txt', 'a')
# fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n'\
#           %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, 
#             J_isco/S_Mt, L_isco/S_Mt, e_isco, 
#             Sm_isco/S_Mt, Sp_isco/S_Mt))
# fid.close()

# t_run1 = timeit.default_timer()
# print('dJdL run (to 6 M):', t_run1 - t_run0)

#######################################################################
### plot some results for debugging
#######################################################################

# if plot_flag>0:
#     ## plot the dJ/dL evolution tracks
#     LL, JJ = sol.t*S_Mt, sol.y[0, :]*S_Mt
#     ee = e_vs_L_func(LL)
#     aa = LL**2./(mu_i**2.*G*(M1+M2)*(1.-ee**2.))
#     nLL = len(LL)

#     dJdL, dJdLm, dJdLp = np.zeros(nLL), np.zeros(nLL), np.zeros(nLL)
#     t_pre_m, t_pre_p = np.zeros(nLL), np.zeros(nLL)
#     t_gw = np.zeros(nLL)
#     for i in range(nLL):
#         dJdL[i] = int_func(LL[i]/S_Mt, JJ[i]/S_Mt)
    
#         SSm, SSp = LK.find_Smp(JJ[i], LL[i], ee[i], par_JL, nPt)
#         dJdLm[i] = (JJ[i]**2. + LL[i]**2. - SSm**2.)/(2.*JJ[i]*LL[i])
#         dJdLp[i] = (JJ[i]**2. + LL[i]**2. - SSp**2.)/(2.*JJ[i]*LL[i])
    
#         dSdt_m = np.abs(LK.get_dSdt(JJ[i], LL[i], ee[i], SSm, par_JL))
#         dSdt_p = np.abs(LK.get_dSdt(JJ[i], LL[i], ee[i], SSp, par_JL))
    
#         t_pre_m[i] = SSm/dSdt_m
#         t_pre_p[i] = SSp/dSdt_p
#         t_gw[i]=LK.get_inst_t_gw_from_a_orb(M1, M2, aa[i], ee[i])
    
#     fig=plt.figure()
#     ax=fig.add_subplot(211)
#     ax.plot(aa/r_Mt, dJdL, label=r'$<\frac{{\rm d}J}{{\rm d}L}>$')
#     ax.fill_between(aa/r_Mt, dJdLp, dJdLm, color='tab:grey', alpha=0.5)
#     ax.set_ylabel(r'\frac{{\rm d}J}{{\rm d}L}')
#     ax.set_xscale('log')

#     ax=fig.add_subplot(212)
#     ax.plot(aa/r_Mt, (dJdLm-dJdLp)/dJdL)
#     ax.set_xscale('log')
#     ax.set_ylabel('Fractional Variation')
#     ax.set_xlabel(r'$a/M_{\rm t}$')
#     fig.savefig(fig_dir + prefix + 'dJdL_evol.pdf')
#     plt.close()

#     fig=plt.figure()
#     ax=fig.add_subplot(111)
#     ax.loglog(aa/r_Mt, t_gw/P_yr, label=r'$t_{\rm gw}$')
#     ax.loglog(aa/r_Mt, t_pre_m/P_yr, label=r'$S/({\rm d}S/{\rm d}t)|_{S_-}$')
#     ax.loglog(aa/r_Mt, t_pre_p/P_yr, label=r'$S/({\rm d}S/{\rm d}t)|_{S_+}$')
#     ax.set_ylim([0.03*np.min(t_gw)/P_yr, 30.*np.max(t_gw)/P_yr])
#     ax.legend(loc='upper left')
#     ax.set_xlabel(r'$a/M_{\rm t}$')
#     ax.set_ylabel(r'Timescales [yr]')
            
#     ax=ax.twinx()
#     ax.loglog(aa/r_Mt, t_gw/(r_Mt/c), ls='')
#     ax.loglog(aa/r_Mt, t_pre_m/(r_Mt/c), ls='')
#     ax.loglog(aa/r_Mt, t_pre_p/(r_Mt/c), ls='')
#     ax.set_ylim([0.03*np.min(t_gw)/(r_Mt/c), 30.*np.max(t_gw)/(r_Mt/c)])
#     ax.set_ylabel(r'$t/M_{\rm t}$')
#     fig.savefig(fig_dir + prefix + 'timescales.pdf')
#     plt.close()

#     ## plot the contours
#     S_LK, chi1_LK, chi2_LK = LK.find_S_chi_contour(J_LK, L_LK, e_LK, par_JL)
#     S_600, chi1_600, chi2_600 = LK.find_S_chi_contour(J_600, L_600, e_600, par_JL)
#     S_isco, chi1_isco, chi2_isco = LK.find_S_chi_contour(J_isco, L_isco, e_isco, par_JL)

#     fig=plt.figure()
#     ax=fig.add_subplot(111)

#     ax.plot(S_LK/S_Mt, chi1_LK, color='tab:blue', alpha=0.8, label=r'$a=a_0/30$')
#     ax.plot(S_LK/S_Mt, chi2_LK, color='tab:blue', alpha=0.8)
#     ax.plot(S_600/S_Mt, chi1_600, color='tab:orange', alpha=0.8, label=r'$a=10^3 M_{\rm t}$')
#     ax.plot(S_600/S_Mt, chi2_600, color='tab:orange', alpha=0.8)
#     ax.plot(S_isco/S_Mt, chi1_isco, color='tab:green', alpha=0.8, label=r'$a=6 M_{\rm t}$')
#     ax.plot(S_isco/S_Mt, chi2_isco, color='tab:green', alpha=0.8)
#     ax.axhline(chi_eff, color='tab:grey', ls=':', alpha=0.7, label=r'True value of $\chi_{\rm eff}$')

#     ax.legend()
#     ax.set_ylabel(r'$\chi_{\rm eff}$')
#     ax.set_xlabel(r'$S/M_{\rm t}^2$')

#     fig.savefig(fig_dir + prefix + 'S_chi_contour.pdf')
#     plt.close()

