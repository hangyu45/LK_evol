#!/usr/bin/env python

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
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
                     'xtick.labelsize': 'medium',
                     'ytick.labelsize': 'medium',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.73,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 17,
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'savefig.dpi': 80,
                     'pdf.compression': 9})

from myConstants import *
import LKlib as LK

#######################################################################
### input 
#######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', type=np.int, default=0)
parser.add_argument('--M3', type=np.float, default=30., 
                   help='M3 in [Msun]')
parser.add_argument('--ao', type=np.float, default=0.021817, 
                   help='Semi-major axis of the outer orbit in [pc]')
parser.add_argument('--ai-0', type=np.float, default=100., 
                   help='Initial semi-major axis of the inner orbit in [AU]')
parser.add_argument('--atol', type=np.float, default=1e-9)
parser.add_argument('--rtol', type=np.float, default=1e-9)

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)

run_id=kwargs['run_id']
M3, ao, ai_0=kwargs['M3'], kwargs['ao'], kwargs['ai_0']
atol, rtol = kwargs['atol'], kwargs['rtol']
br_flag, ss_flag=1, 1

fig_dir = '/home/hang.yu/public_html/astro/LK_evol/LK2merger/M3_%.1e_ao_%.1epc_ai0_%.1eAU/'\
            %(M3, ao, ai_0)
data_dir = 'data/LK2merger/M3_%.1e_ao_%.1epc_ai0_%.1eAU/'\
            %(M3, ao, ai_0)
prefix = 'id_%i_'%(run_id)
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
M1, M2 = 30.*Ms, 20.*Ms
chi1, chi2=0.7, 0.7
I_io_0 = 92.52*np.pi/180.
I_S1_0 = 92.52*np.pi/180.
phi_S1_0 = 45.*np.pi/180.
I_S2_0 = 92.52*np.pi/180.
phi_S2_0 = 0.*np.pi/180.

## record initial condition
fid = open(data_dir + prefix + 'init_cond.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, 
            I_io_0, \
            I_S1_0, phi_S1_0, \
            I_S2_0, phi_S2_0))
fid.close()

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

t_unit=t_LK_0
Li_unit, Lo_unit = Li_0, Lo
ai_unit = ai_0
S1_unit = S1
S2_unit = S2
par_LK = np.array([M1, M2, M3, ao, \
        t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
        br_flag, ss_flag])

ai_nat = ai_0/ai_unit
Li_nat_v = np.array([np.sin(I_io_0), 0., np.cos(I_io_0)]) * Li_0/Li_unit
ei_v = np.array([0., ei_0, 0])
Lo_nat_v = np.array([0., 0., 1.])*Lo/Lo_unit
eo_v = np.zeros(3)
S1_nat_v = np.array([np.sin(I_S1_0)*np.cos(phi_S1_0), 
                     np.sin(I_S1_0)*np.sin(phi_S1_0), 
                     np.cos(I_S1_0)]) * S1/S1_unit
S2_nat_v = np.array([np.sin(I_S2_0)*np.cos(phi_S2_0), 
                     np.sin(I_S2_0)*np.sin(phi_S2_0),               
                     np.cos(I_S2_0)]) * S2/S2_unit

y_nat_init = np.hstack([
    ai_nat, \
    Li_nat_v, ei_v, \
    Lo_nat_v, eo_v, \
    S1_nat_v, S2_nat_v])

#######################################################################
### solve LK ai = ai_0/30
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
        t_span=(0, 1e4), y0=y_nat_init, rtol=rtol, atol=atol, \
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

# Lo_x = sol.y[7, :]*Lo_unit 
# Lo_y = sol.y[8, :]*Lo_unit 
# Lo_z = sol.y[9, :]*Lo_unit 
# Lo = np.sqrt(Lo_x**2. + Lo_y**2. + Lo_z**2.)
# eo_x = sol.y[10, :]
# eo_y = sol.y[11, :]
# eo_z = sol.y[12, :]
# eo=np.sqrt(eo_x**2. + eo_y**2. + eo_z**2.)

S1_x = sol.y[13, :]*S1_unit
S1_y = sol.y[14, :]*S1_unit
S1_z = sol.y[15, :]*S1_unit
S1 = np.median(np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.))

S2_x = sol.y[16, :]*S2_unit 
S2_y = sol.y[17, :]*S2_unit
S2_z = sol.y[18, :]*S2_unit
S2 = np.median(np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.))

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

chi_eff = np.median((M1*uS1_d_uLi*chi1 + M2*uS2_d_uLi*chi2)/(M1+M2))

theta1_SL = np.arccos(uS1_d_uLi)
theta2_SL = np.arccos(uS2_d_uLi)
theta_SS = np.arccos(uS1_d_uS2)

## record condition at the end of LK
fid = open(data_dir + prefix + 'LK_cond.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6f\t%.6f\t%.6f\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, 
            ai[-1]/r_Mt, ei[-1], 
            J[-1]/S_Mt, Li[-1]/S_Mt, S[-1]/S_Mt, 
            theta1_SL[-1], theta2_SL[-1], theta_SS[-1]))
fid.close()

fig=plt.figure(figsize=(9, 14))
ax=fig.add_subplot(411)
ax.semilogy(tt/1.e9/P_yr, ai/AU, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$a_{\rm in}$ [AU]')

ax=fig.add_subplot(412)
ax.semilogy(tt/1.e9/P_yr, 1.-ei, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$1-e_{\rm in}$')

ax=fig.add_subplot(413)
ax.plot(tt/1.e9/P_yr, theta1_SL*180./np.pi, alpha=0.8, label=r'$S_1$', color='tab:grey')
ax.plot(tt/1.e9/P_yr, theta2_SL*180./np.pi, alpha=0.8, label=r'$S_2$', color='tab:olive')
ax.axhline(90., color='tab:red', ls=':', alpha=0.5)
ax.set_ylabel(r'$\theta_{SL}$ [$^\circ$]')
ax.legend(loc='upper left')

ax=fig.add_subplot(414)
ax.plot(tt/1.e9/P_yr, theta_SS*180./np.pi, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$\theta_{SS}$ [$^\circ$]')
ax.set_xlabel(r'Time [Gyr]')

plt.subplots_adjust(hspace=0)
plt.savefig(fig_dir+prefix+'LK_evol.pdf')
plt.close()


#######################################################################
### evolve (scalars) a, L, e to merger
#######################################################################

## quantities at the end of the LK run
a_LK=ai[-1]
L_LK=Li[-1]
J_LK=J[-1]

# e_LK=ei[-1]
eff_LK = L_LK/(mu_i*np.sqrt(G*(M1+M2)*a_LK))
e_LK = np.sqrt(1. - eff_LK**2.)

t_GW_LK = LK.get_inst_t_gw_from_a_orb(M1, M2, a_LK, e_LK) 
print('t_GW when a=a0/10: [yr]', t_GW_LK/P_yr)
print('a/AU, e, e/e_ref', a_LK/AU, e_LK, e_LK/ei[-1])
print('chi_eff', chi_eff)
print('L/Mt^2, J/Mt^2', L_LK/S_Mt, J_LK/S_Mt)

par_aL = np.array([M1, M2])

int_func=lambda loga_, logL_:\
    LK.evol_logL_vs_loga(loga_, logL_, par_aL)
    
sol=integ.solve_ivp(int_func, \
        t_span=(np.log(a_LK/r_Mt), np.log(6.)), y0=np.array([np.log(L_LK/S_Mt)]), rtol=3e-14, atol=1e-12)

a_scal = np.exp(sol.t)*r_Mt
L_scal = np.exp(sol.y[0, :])*S_Mt
eff_scal = L_scal/(mu_i*np.sqrt(G*(M1+M2)*a_scal))
e_scal = np.sqrt(1. - eff_scal**2.)

print(L_scal[-1]/S_Mt)

loge_vs_logL_tck = interp.splrep(np.log(L_scal[::-1]), np.log(e_scal[::-1]))
logL_vs_loga_tck = interp.splrep(np.log(a_scal[::-1]), np.log(L_scal[::-1]))
e_vs_L_func=lambda LL: np.exp(interp.splev(np.log(LL), loge_vs_logL_tck))
L_vs_a_func=lambda aa: np.exp(interp.splev(np.log(aa), logL_vs_loga_tck))

#######################################################################
### J vs L func
#######################################################################

nPt = 100
par_JL = np.array([M1, M2, S1, S2, chi_eff])
int_func=lambda L_nat_, J_nat_:\
    LK.evol_J_avg(L_nat_, J_nat_, e_vs_L_func, par_JL, nPt=nPt)

#######################################################################
### evol to 1000 r_Mt
#######################################################################

t_run0 = timeit.default_timer()

L_1kM = L_vs_a_func(1000.*r_Mt)
e_1kM = e_vs_L_func(L_1kM)
sol=integ.solve_ivp(int_func, \
    t_span=(L_LK/S_Mt, L_1kM/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-12, atol=3e-12)

J_1kM = sol.y[0,-1] * S_Mt
print('r = 1000 M')
print('J, L, e', J_1kM/S_Mt, L_1kM/S_Mt, e_1kM)

Sm_1kM, Sp_1kM=LK.find_Smp(J_1kM, L_1kM, e_1kM, par_JL)

fid = open(data_dir + prefix + 'r_1kM_cond.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, 
            J_1kM/S_Mt, L_1kM/S_Mt, e_1kM, 
            Sm_1kM/S_Mt, Sp_1kM/S_Mt))
fid.close()

t_run1 = timeit.default_timer()
print('dJdL run (to 1000 M):', t_run1 - t_run0)

#######################################################################
### evol to ISCO
#######################################################################

t_run0 = timeit.default_timer()

L_isco = L_vs_a_func(6.*r_Mt)
e_isco = e_vs_L_func(L_isco)
sol=integ.solve_ivp(int_func, \
    t_span=(L_LK/S_Mt, L_isco/S_Mt), y0=np.array([J_LK/S_Mt]), rtol=3e-13, atol=1e-14)

J_isco = sol.y[0,-1] * S_Mt
print('r = 6 M')
print('J, L, e', J_isco/S_Mt, L_isco/S_Mt, e_isco)

Sm_isco, Sp_isco=LK.find_Smp(J_isco, L_isco, e_isco, par_JL)

fid = open(data_dir + prefix + 'r_isco_cond.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, 
            J_isco/S_Mt, L_isco/S_Mt, e_isco, 
            Sm_isco/S_Mt, Sp_isco/S_Mt))
fid.close()

t_run1 = timeit.default_timer()
print('dJdL run (to 6 M):', t_run1 - t_run0)

## plot the dJ/dL evolution tracks
LL, JJ = sol.t*S_Mt, sol.y[0, :]*S_Mt
ee = e_vs_L_func(LL)
aa = LL**2./(mu_i**2.*G*(M1+M2)*(1.-ee**2.))
nLL = len(LL)

dJdL, dJdLm, dJdLp = np.zeros(nLL), np.zeros(nLL), np.zeros(nLL)
for i in range(nLL):
    dJdL[i] = int_func(LL[i]/S_Mt, JJ[i]/S_Mt)
    
    SSm, SSp = LK.find_Smp(JJ[i], LL[i], e_vs_L_func(LL[i]), par_JL, nPt)
    dJdLm[i] = (JJ[i]**2. + LL[i]**2. - SSm**2.)/(2.*JJ[i]*LL[i])
    dJdLp[i] = (JJ[i]**2. + LL[i]**2. - SSp**2.)/(2.*JJ[i]*LL[i])
    
fig=plt.figure()
ax=fig.add_subplot(211)
ax.plot(aa/r_Mt, dJdL, label=r'$<\frac{{\rm d}J}{{\rm d}L}>$')
ax.fill_between(aa/r_Mt, dJdLp, dJdLm, color='tab:grey', alpha=0.5)
ax.set_ylabel(r'\frac{{\rm d}J}{{\rm d}L}')
ax.set_xscale('log')

ax=fig.add_subplot(212)
ax.plot(aa/r_Mt, (dJdLm-dJdLp)/dJdL)
ax.set_xscale('log')
ax.set_ylabel('Fractional Variation')
ax.set_xlabel(r'$a/M_{\rm t}$')
plt.savefig(fig_dir + prefix + 'dJdL_evol.pdf')
plt.close()

## plot the contours
S_LK, chi1_LK, chi2_LK = LK.find_S_chi_contour(J_LK, L_LK, e_LK, par_JL)
S_1kM, chi1_1kM, chi2_1kM = LK.find_S_chi_contour(J_1kM, L_1kM, e_1kM, par_JL)
S_isco, chi1_isco, chi2_isco = LK.find_S_chi_contour(J_isco, L_isco, e_isco, par_JL)

fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(S_LK/S_Mt, chi1_LK, color='tab:blue', alpha=0.8, label=r'$a=a_0/30$')
ax.plot(S_LK/S_Mt, chi2_LK, color='tab:blue', alpha=0.8)
ax.plot(S_1kM/S_Mt, chi1_1kM, color='tab:orange', alpha=0.8, label=r'$a=10^3 M_{\rm t}$')
ax.plot(S_1kM/S_Mt, chi2_1kM, color='tab:orange', alpha=0.8)
ax.plot(S_isco/S_Mt, chi1_isco, color='tab:green', alpha=0.8, label=r'$a=6 M_{\rm t}$')
ax.plot(S_isco/S_Mt, chi2_isco, color='tab:green', alpha=0.8)
ax.axhline(chi_eff, color='tab:grey', ls=':', alpha=0.7, label=r'True value of $\chi_{\rm eff}$')

ax.legend()
ax.set_ylabel(r'$\chi_{\rm eff}$')
ax.set_xlabel(r'$S/M_{\rm t}^2$')

plt.savefig(fig_dir + prefix + 'S_chi_contour.pdf')
plt.close()

