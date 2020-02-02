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
import os

from numba import jit, prange

plt.rc('figure', figsize=(9, 7))
plt.rcParams.update({'text.usetex': False,
                     'font.family': 'serif',
                     'font.serif': ['Georgia'],
                     'mathtext.fontset': 'cm',
                     'lines.linewidth': 2.5,
                     'font.size': 18,
                     'xtick.labelsize': 'medium',
                     'ytick.labelsize': 'medium',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
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

###################################################
### input 
###################################################
out_dir = '/home/hang.yu/public_html/LK_evol/'
out_base = 'binary_'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

M1, M2= 30.*Ms, 20.*Ms
mu_i = M1*M2/(M1+M2)

ai_0 = 10.*AU

br_flag, ss_flag = 1, 1

ei_0 = 1.-1.e-3
eff_i_0 = np.sqrt(1.-ei_0**2.)

Li_0 = mu_i * np.sqrt(G*(M1+M2)*ai_0)*eff_i_0

chi_1, chi_2 = 0.1, 0.1
S1 = chi_1 * G*M1**2./c
S2 = chi_2 * G*M2**2./c

omega_i_0 = np.sqrt(G*(M1+M2)/ai_0**3.)
print(2.*np.pi/omega_i_0/P_yr)

t_GW_0 = LK.get_inst_t_gw_from_a_orb(M1, M2, ai_0, ei_0)
print('%e'%(t_GW_0/P_yr))

t_unit=1.e-2*t_GW_0
Li_unit = Li_0
ai_unit = AU
S1_unit = S1
S2_unit = S2

par = np.array([t_unit, Li_unit, 1, ai_unit, S1_unit, S2_unit, \
                M1, M2, 1, 1, \
                br_flag, ss_flag])

Li_nat_x, Li_nat_y, Li_nat_z = 0, 0, Li_0/Li_unit
ei_x, ei_y, ei_z = ei_0, 0., 0.
ai_nat = ai_0/ai_unit
S1_nat_x, S1_nat_y, S1_nat_z = S1/S1_unit, 0., 0.
S2_nat_x, S2_nat_y, S2_nat_z = 0., S2/S2_unit, 0.

y_nat_init = np.array([
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    ai_nat, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                ])

print(LK.evol_binary(0, y_nat_init, par))
###################################################


###################################################
### solve ode 
###################################################
def terminator(t_nat, y_nat_vect, par):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    ai_nat, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    M1, M2, __, __, \
    br_flag, ss_flag\
                = par
        
    ai = ai_nat * ai_unit
    resi = ai - 1.e-4*AU
    resi /= AU
    return resi

term_func=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par)
term_func.direction = -1
term_func.terminal = True

int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_binary(t_nat_, y_nat_vect_, par)
    
sol=integ.solve_ivp(int_func, \
        t_span=(0, 1e4), y0=y_nat_init, rtol=3e-8, atol=3e-8, \
        events=term_func)
###################################################

###################################################
### plot results
###################################################

tt = sol.t*t_unit
print('number of time steps,', len(tt))

Li_x = sol.y[0, :]*Li_unit 
Li_y = sol.y[1, :]*Li_unit 
Li_z = sol.y[2, :]*Li_unit 
Li = np.sqrt(Li_x**2. + Li_y**2. + Li_z**2.)

ei_x = sol.y[3, :]
ei_y = sol.y[4, :]
ei_z = sol.y[5, :]
ei=np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)

ai = sol.y[6, :]*ai_unit 

S1_x = sol.y[7, :]*S1_unit
S1_y = sol.y[8, :]*S1_unit
S1_z = sol.y[9, :]*S1_unit
S1 = np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.)
S2_x = sol.y[10, :]*S2_unit 
S2_y = sol.y[11, :]*S2_unit
S2_z = sol.y[12, :]*S2_unit
S2 = np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.)

theta_SL_1 = np.real(np.arccos(\
        (S1_x*Li_x + S1_y*Li_y + S1_z*Li_z)/(S1*Li)+0j
                      ))
theta_SL_2 = np.real(np.arccos(\
        (S2_x*Li_x + S2_y*Li_y + S2_z*Li_z)/(S2*Li)+0j
                      ))

theta_SS = np.real(np.arccos(\
        (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)+0j
                    ))

fig=plt.figure(figsize=(9, 14))
ax=fig.add_subplot(411)
ax.semilogy(tt/P_yr, ai/AU, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$a_{\rm in}$ [AU]')

ax=fig.add_subplot(412)
ax.semilogy(tt/P_yr, 1.-ei, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$1-e_{\rm in}$')

ax=fig.add_subplot(413)
ax.plot(tt/P_yr, theta_SL_1*180./np.pi, alpha=0.8, label=r'$S_1$', color='tab:grey')
ax.plot(tt/P_yr, theta_SL_2*180./np.pi, alpha=0.8, label=r'$S_2$', color='tab:olive')
ax.axhline(90., color='tab:red', ls=':', alpha=0.5)
ax.set_ylabel(r'$\theta_{SL}$ [deg]')
ax.legend(loc='upper left')

ax=fig.add_subplot(414)
ax.plot(tt/P_yr, theta_SS*180./np.pi, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$\theta_{SS}$ [deg]')
ax.set_xlabel(r'Time [yr]')
plt.subplots_adjust(hspace=0)
plt.savefig(out_dir + out_base + 'evol_traj.pdf')

nLast=100
fig=plt.figure()
ax=fig.add_subplot(211)
ax.plot(tt[-nLast:]/3600, theta_SL_1[-nLast:]*180./np.pi, alpha=0.8, label=r'$S_1$', color='tab:grey')
ax.plot(tt[-nLast:]/3600, theta_SL_2[-nLast:]*180./np.pi, alpha=0.8, label=r'$S_2$', color='tab:olive')
ax.axhline(90., color='tab:red', ls=':', alpha=0.5)
ax.set_ylabel(r'$\theta_{SL}$ [deg]')
ax.legend()

ax=fig.add_subplot(212)
ax.plot(tt[-nLast:]/3600, theta_SS[-nLast:]*180./np.pi, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$\theta_{SS}$ [deg]')
ax.set_xlabel(r'Time [hr]')
plt.subplots_adjust(hspace=0)
plt.savefig(out_dir + out_base + 'evol_traj_bin.pdf')