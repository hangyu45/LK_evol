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
parser.add_argument('--a-0', type=np.float, default=300., 
                   help='Initial semi-major axis of the inner orbit in [r_Mt]')
parser.add_argument('--atol', type=np.float, default=0.3e-10)
parser.add_argument('--rtol', type=np.float, default=0.3e-10)
parser.add_argument('--plot-flag', type=np.int, default=1)
parser.add_argument('--fudge-gw', type=np.float, default=1.)

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)


run_id=kwargs['run_id']
a_0=kwargs['a_0']
atol, rtol = kwargs['atol'], kwargs['rtol']
br_flag, ss_flag=1, 1
plot_flag = kwargs['plot_flag']
fudge_gw = kwargs['fudge_gw']

fig_dir = '/home/hang.yu/public_html/astro/LK_evol/bin_evol/a0_%.1eAU/'\
            %(a_0)
data_dir = 'data/bin_evol/a0_%.1eAU/'\
            %(a_0)
prefix = 'id_%i_gw_%.1f_'%(run_id, fudge_gw)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#######################################################################
### MCMC
#######################################################################

## FIXME ##
# fix M1, M2, chi1, chi2
M1 = 85.*Ms
M2 = 65.*Ms

chi1, chi2 = 0.7, 0.7
I_S1_0 = 75.*np.pi/180. #np.pi/2. #stats.uniform(scale=np.pi).rvs() 
phi_S1_0 = 0. #stats.uniform(scale=2.*np.pi).rvs()
I_S2_0 = 110.*np.pi/180. #np.pi/2. #stats.uniform(scale=np.pi).rvs()
phi_S2_0 = 107.*np.pi/180. #stats.uniform(scale=2.*np.pi).rvs()

# ### Hacking!!! ###
# I_S1_0, I_S2_0=np.pi/2., np.pi/2.

chi1_x, chi1_y, chi1_z = \
    chi1*np.sin(I_S1_0)*np.cos(phi_S1_0), \
    chi1*np.sin(I_S1_0)*np.sin(phi_S1_0), \
    chi1*np.cos(I_S1_0)

chi2_x, chi2_y, chi2_z = \
    chi2*np.sin(I_S2_0)*np.cos(phi_S2_0), \
    chi2*np.sin(I_S2_0)*np.sin(phi_S2_0), \
    chi2*np.cos(I_S2_0)

chi_eff = (M1*chi1_z+M2*chi2_z)/(M1+M2)

uS1_0 = np.array([chi1_x, chi1_y, chi1_z])/chi1
uS2_0 = np.array([chi2_x, chi2_y, chi2_z])/chi2
uL_0 = np.array([0, 0, 1])

#######################################################################
### Initial vectors
#######################################################################

mu = M1*M2/(M1+M2)
Mt = M1+M2
qq = M2/M1
eta = mu/Mt

S1 = chi1 * G*M1**2./c
S2 = chi2 * G*M2**2./c

r_Mt=G*(M1+M2)/c**2.
S_Mt=G*(M1+M2)**2./c

a_0 *= r_Mt
e_0 = 0.
eff_0 = np.sqrt(1.-e_0**2.)
L_0 = mu * np.sqrt(G*Mt*a_0)*eff_0

t_GW_0 = LK.get_inst_t_gw_from_a_orb(M1, M2, a_0, e_0)
print('T_gw/yr', '%e'%(t_GW_0/P_yr))
t_GW_isco = LK.get_inst_t_gw_from_a_orb(M1, M2, 6.*r_Mt, 0.)

t_unit=np.sqrt(t_GW_0*t_GW_isco)
L_unit = L_0
a_unit = np.sqrt(a_0*6*r_Mt)
S1_unit = S1
S2_unit = S2
par = np.array([M1, M2, np.inf, np.inf, \
        t_unit, L_unit, np.inf, a_unit, S1_unit, S2_unit, \
        br_flag, ss_flag])

a_nat = a_0/a_unit
L_nat_v = uL_0 * L_0/L_unit
e_v = np.array([0., e_0, 0])
S1_nat_v = uS1_0 * S1/S1_unit
S2_nat_v = uS2_0 * S2/S2_unit

y_nat_init = np.hstack([
    a_nat, \
    L_nat_v, e_v, \
    S1_nat_v, S2_nat_v])

theta_SS_0 = np.arccos(\
        LK.inner(uS1_0, uS2_0))
print('initial theta SS:', theta_SS_0*180./np.pi)

#######################################################################
### solve ode
#######################################################################
t_run0 = timeit.default_timer()
@jit(nopython=True)
def terminator(t_nat, y_nat_vect, par):
    # parse parameters
    ai_nat = y_nat_vect[0]
    M1, M2, __, __, \
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par

    ai = ai_nat*ai_unit
    resi = (ai - 6.*r_Mt)/ai_unit
    return resi

term_func=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par)
term_func.direction = -1
term_func.terminal = True

def isco_finder(t_nat, y_nat_vect, par):
    # parse parameters
    ai_nat = y_nat_vect[0]
    M1, M2, __, __, \
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par

    ai = ai_nat*ai_unit
    resi = (ai - 6.*r_Mt)/ai_unit
    return resi
isco_func=lambda t_nat_, y_nat_vect_:isco_finder(t_nat_, y_nat_vect_, par)
isco_func.terminal = False

int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_binary(t_nat_, y_nat_vect_, par, fudge_gw)
    
sol=integ.solve_ivp(int_func, \
        t_span=(0, 1e9), y0=y_nat_init, rtol=rtol, atol=atol, \
        events=(isco_func, term_func))

t_run1 = timeit.default_timer()
print('run time:', t_run1 - t_run0)

#######################################################################
### output
#######################################################################

# get sol
tt = sol.t*t_unit

a_orb = sol.y[0, :]*a_unit 
f_gw = np.sqrt(G*(M1+M2)/a_orb**3.)/np.pi

L_x = sol.y[1, :]*L_unit 
L_y = sol.y[2, :]*L_unit 
L_z = sol.y[3, :]*L_unit 
L_orb = np.sqrt(L_x**2. + L_y**2. + L_z**2.)

e_x = sol.y[4, :]
e_y = sol.y[5, :]
e_z = sol.y[6, :]
e_orb = np.sqrt(e_x**2. + e_y**2. + e_z**2.)

S1_x = sol.y[7, :]*S1_unit
S1_y = sol.y[8, :]*S1_unit
S1_z = sol.y[9, :]*S1_unit
S1 = np.median(np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.))

S2_x = sol.y[10, :]*S2_unit 
S2_y = sol.y[11, :]*S2_unit
S2_z = sol.y[12, :]*S2_unit
S2 = np.median(np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.))

S_x = S1_x + S2_x
S_y = S1_y + S2_y
S_z = S1_z + S2_z
S = np.sqrt(S_x**2. + S_y**2. + S_z**2.)

J_x = L_x + S1_x + S2_x
J_y = L_y + S1_y + S2_y
J_z = L_z + S1_z + S2_z
J = np.sqrt(J_x**2. + J_y**2. + J_z**2.)

print('J, L:', J[-1]/S_Mt, L_orb[-1]/S_Mt)

theta1_SL = np.real(np.arccos(\
        (S1_x*L_x + S1_y*L_y + S1_z*L_z)/(S1*L_orb)+0j
                      ))
theta2_SL = np.real(np.arccos(\
        (S2_x*L_x + S2_y*L_y + S2_z*L_z)/(S2*L_orb)+0j
                      ))
theta_SS = np.real(np.arccos(\
        (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)+0j
                    ))

chi1_p, chi2_p = np.zeros(len(tt)), np.zeros(len(tt))
chi_eff_num = np.zeros(len(tt))

par_JL = np.array([M1, M2, S1, S2, chi_eff])
tau_gw_inst, tau_pre, tau_pre_inst \
    = np.zeros(len(tt)), np.zeros(len(tt)), np.zeros(len(tt))
tau_pre_min = np.zeros(len(tt)) 
nPt=200

kap = S2/((1-qq)*L_orb)
# kap = (S2*(4+3.*qq)*qq/(3.*(1-qq**2)*L_orb))

c_th_S1L0 = np.cos(theta1_SL[0])

CC = np.cos(theta1_SL) + kap * np.cos(theta_SS) 
# CC *= (1+np.log(L_orb/L_orb[0]))

CC_full = (1.+qq**2.*S_Mt*chi_eff/((1.+qq)**2.*(1.-qq)*L_orb) )*np.cos(theta1_SL)\
        - (qq*S1/2./L_orb/(1.-qq))*np.cos(theta1_SL)**2.+S2/L_orb/(1.-qq) * np.cos(theta_SS)

invqq = 1./qq
CC2_full = (1.+invqq**2.*S_Mt*chi_eff/((1.+invqq)**2.*(1.-invqq)*L_orb) )*np.cos(theta2_SL)\
        - (invqq*S2/2./L_orb/(1.-invqq))*np.cos(theta2_SL)**2.+S1/L_orb/(1.-invqq) * np.cos(theta_SS)

f_gw_isco = np.sqrt(G*Mt/(6.*r_Mt)**3.)/np.pi

for i in range(len(tt)):
#     chi1_p[i]=np.linalg.norm(LK.cross(\
#                        np.array([S1_x[i], S1_y[i], S1_z[i]]), \
#                        np.array([L_x[i],  L_y[i],  L_z[i]]))\
#                      /L_orb[i]/(G*M1**2./c))
#     chi2_p[i]=np.linalg.norm(LK.cross(\
#                        np.array([S2_x[i], S2_y[i], S2_z[i]]), \
#                        np.array([L_x[i],  L_y[i],  L_z[i]]))\
#                      /L_orb[i]/(G*M2**2./c))
#     chi_eff_num[i] = (M1*LK.inner(np.array([S1_x[i], S1_y[i], S1_z[i]]), \
#                                   np.array([L_x[i],  L_y[i],  L_z[i] ]))\
#                               /L_orb[i]/(G*M1**2./c)\
#                      +M2*LK.inner(np.array([S2_x[i], S2_y[i], S2_z[i]]), \
#                                   np.array([L_x[i],  L_y[i],  L_z[i] ]))\
#                               /L_orb[i]/(G*M2**2./c))\
#                     /(M1+M2)
    
#     omega_ss = 1.5 * G*(M1+M2+2.*mu/3.)/(c**2.*a_orb)*np.pi*f_gw
    
    tau_gw_inst[i] = LK.get_inst_t_gw_from_a_orb(M1, M2, a_orb[i], 0)/fudge_gw
    
    Sm, Sp = LK.find_Smp(J[i], L_orb[i], e_orb[i], par_JL, nPt=nPt)
    S_vect = np.linspace(Sm, Sp, nPt)
#     dSdt = np.zeros(nPt)
#     for j in range(nPt):
#         dSdt[j] = LK.get_dSdt(J[i], L_orb[i], e_orb[i], S_vect[j], par_JL)

    dSdt = LK.get_dSdt(J[i], L_orb[i], e_orb[i], S_vect, par_JL)
    
    idx=np.isfinite(dSdt)
        
    tau_pre[i] = 2.*integ.trapz(1./np.abs(dSdt[idx]), S_vect[idx])
    tau_pre_min[i] = np.min((S1+S2)/np.abs(dSdt))
    
#     tau_pre[i] = LK.get_tau_pre(J[i], L_orb[i], e_orb[i], par_JL, nPt=100)
    tau_pre_inst[i] = (S1+S2)/np.abs(LK.get_dSdt(J[i], L_orb[i], e_orb[i], S[i], par_JL))

# fid = open(data_dir + prefix + 'bin_evol.txt', 'a')
# fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'\
#           %(M1/Ms, M2/Ms, \
#             chi1_x, chi1_y, chi1_z, \
#             chi2_x, chi2_y, chi2_z, \
#             theta_SS_0, \
#             theta1_SL[-1], theta2_SL[-1], theta_SS[-1]))
# fid.close()

if plot_flag:
    fig=plt.figure(figsize=(9, 14))
    ax=fig.add_subplot(311)
    ax.semilogx(f_gw, theta_SS*180./np.pi, label=r'$\theta_{S_1S_2}$', alpha=0.75, color='tab:grey')
    ax.semilogx(f_gw, theta1_SL*180./np.pi, label=r'$\theta_{S_1 L}$', alpha=0.75, color='tab:olive')
    ax.semilogx(f_gw, theta2_SL*180./np.pi, label=r'$\theta_{S_2 L}$', alpha=0.75, color='tab:cyan')
    
#     ax.semilogx(f_gw, 90.+10.*np.cos(omega_ss*(tt[-1]-tt)), alpha=0.5)
    ax.legend(loc='lower left')
    ax.set_ylabel(r'$\theta$ [$^\circ$]')
#     ax.invert_xaxis()
    ax.set_xticklabels([])

    
    
    ax=fig.add_subplot(312)
    ax.semilogx(f_gw, CC, \
                alpha=0.75, label=r'$\cos\theta_{S_1L} + \left[S_2/(1-q) L\right]\cos\theta_{S_1S_2}$', color='tab:purple', zorder=10)
    
    ax.semilogx(f_gw, CC_full, \
                alpha=0.75, label=r'$\mathcal{C}$', color='tab:red', zorder=11)
    
#     ax.semilogx(f_gw, CC2_full, \
#                 alpha=0.75, label=r'Full 2', color='tab:green')
    
    ax.semilogx(f_gw, np.cos(theta1_SL), alpha=0.5, ls='-', \
                label=r'$\cos \theta_{S_1L}$', color='tab:olive')
    ax.semilogx(f_gw, np.cos(theta_SS), alpha=0.5, ls='-',\
                label=r'$\cos \theta_{S_1S_2}$', color='tab:grey')
#     ax.semilogx(f_gw, (np.cos(theta2_SL)-qq**2.*np.cos(theta_SS))/(1.+qq**2.), alpha=0.5, \
#                label=r'$(\cos\theta_{S_2L} + \cos\theta_{S_1S_2})/2$')
#     ax.axhline(np.cos(theta1_SL[0]), ls=':', color='tab:red', alpha=0.75, \
#                label=r'$\cos \theta_{S_1 L}^{(0)}$')
#     ax.axhline(np.cos(theta2_SL[0]), ls='--', color='tab:orange', alpha=0.5, \
#                label=r'$\cos \theta_{S_2 L}^{(0)}$')
    ax.set_ylabel(r'$\cos \theta$')
    ax.legend(loc='lower left')
#     ax.invert_xaxis()

    
    ax=fig.add_subplot(313)
    
#     ax.semilogx(f_gw, chi_eff_num, label=r'$\chi_{\rm eff}$')
#     ax.semilogx(f_gw, chi1_p, label=r'$\chi_{1,\bot}$')
#     ax.semilogx(f_gw, chi2_p, label=r'$\chi_{2,\bot}$')

    ax.loglog(f_gw, tau_gw_inst, label=r'$\tau_{\rm gw}$', color='tab:grey', alpha=0.6)
    ax.loglog(f_gw, tau_pre, label=r'$\tau_{\rm pre}$', ls='-', alpha=0.6, color='tab:olive')
#     ax.loglog(f_gw, tau_pre_inst, label=r'$\tau_{\rm pre, ins}$', ls='--', alpha=0.5, color='tab:cyan')
#     ax.loglog(f_gw, tau_pre_min, label=r'$\tau_{\rm pre, min}$', ls='-.', alpha=0.5, color='tab:purple')
    ax.legend(loc='lower left')
    ax.set_ylabel(r'$\tau$ [s]')
    ax.set_xlabel(r'$f_{\rm gw}$ [Hz]')
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels([0.1, 1, 10])
    
    plt.subplots_adjust(hspace=0)
    plt.savefig(fig_dir + prefix + 'spin_bin.pdf')
    plt.close()
    