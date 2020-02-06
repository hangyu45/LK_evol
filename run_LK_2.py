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
import os, sys

from numba import jit, prange
import timeit

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

#######################################################################
### load input 
#######################################################################
# now the inner binary's orbit is 250x smaller than its initial value
# decouple w/ the third body and evol the inner binary only
# evolve to 30 days before merger

print('run 2')
t_run0 = timeit.default_timer()

fig_dir = '/home/hang.yu/public_html/astro/LK_evol/test/'
data_dir = 'data/'
past_prefix = 'seg1_'
prefix = 'seg2_'

br_flag, ss_flag = 1, 1
atol, rtol = 1e-9, 1e-9

fid=h5.File(data_dir + past_prefix + 'data.h5', 'r')
M1, M2= fid.attrs['M1'], fid.attrs['M2']
mu_i=M1*M2/(M1+M2)
chi1, chi2 = fid.attrs['chi1'], fid.attrs['chi2']

t_0 = fid['t'][()][-1]
ai_0 = fid['ai'][()][-1]

Li_x_0, Li_y_0, Li_z_0 = fid['Li_x'][()][-1], fid['Li_y'][()][-1], fid['Li_z'][()][-1]
Li_0 = np.sqrt(Li_x_0**2. + Li_y_0**2. + Li_z_0**2.)

ei_x_0, ei_y_0, ei_z_0 = fid['ei_x'][()][-1], fid['ei_y'][()][-1], fid['ei_z'][()][-1]
ei_0 = np.sqrt(ei_x_0**2. + ei_y_0**2. + ei_z_0**2.)
eff_i_0 = np.sqrt(1.-ei_0**2.)

S1_x_0, S1_y_0, S1_z_0 = fid['S1_x'][()][-1], fid['S1_y'][()][-1], fid['S1_z'][()][-1]
S1_0 = np.sqrt(S1_x_0**2. + S1_y_0**2. + S1_z_0**2.)

S2_x_0, S2_y_0, S2_z_0 = fid['S2_x'][()][-1], fid['S2_y'][()][-1], fid['S2_z'][()][-1]
S2_0 = np.sqrt(S2_x_0**2. + S2_y_0**2. + S2_z_0**2.)

fid.close()

print('ai/AU, ei', ai_0/AU, ei_0)
print(Li_0/(mu_i*np.sqrt(G*(M1+M2)*ai_0)*eff_i_0))
t_GW_0 = LK.get_inst_t_gw_from_a_orb(M1, M2, ai_0, ei_0)
print('T_gw/yr:', '%e'%(t_GW_0/P_yr))

t_unit = 3e-3 * t_GW_0
Li_unit = Li_0
ai_unit = 0.1*ai_0
S1_unit = S1_0
S2_unit = S2_0

par = np.array([M1, M2, 1, 1, \
                t_unit, Li_unit, 1, ai_unit, S1_unit, S2_unit, \
                br_flag, ss_flag])

ai_nat = ai_0/ai_unit
Li_nat_v = np.array([Li_x_0, Li_y_0, Li_z_0])/Li_unit
ei_v = np.array([ei_x_0, ei_y_0, ei_z_0])
S1_nat_v = np.array([S1_x_0, S1_y_0, S1_z_0])/S1_unit
S2_nat_v = np.array([S2_x_0, S2_y_0, S2_z_0])/S2_unit

y_nat_init = np.hstack([
    ai_nat, \
    Li_nat_v, ei_v, \
    S1_nat_v, S2_nat_v])
#######################################################################

#######################################################################
### solve ode
#######################################################################
@jit(nopython=True)
def terminator(t_nat, y_nat_vect, par):
    # parse parameters
    ai_nat,\
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    M1, M2, __, __, \
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par

    ai = ai_nat*ai_unit
    ei = np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)
    t_gw = LK.get_inst_t_gw_from_a_orb(M1, M2, ai, ei)
    resi = (t_gw - 30.*P_day)
    return resi

term_func=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par)
term_func.direction = -1
term_func.terminal = True

int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_binary(t_nat_, y_nat_vect_, par)
    
sol=integ.solve_ivp(int_func, \
        t_span=(t_0/t_unit, t_0/t_unit+1e5), y0=y_nat_init, rtol=rtol, atol=atol, \
        events=term_func)
#######################################################################

#######################################################################
### output
#######################################################################
t_run1 = timeit.default_timer()
print('run time:', t_run1 - t_run0)

# get sol
tt = sol.t*t_unit
tt-=tt[0]

ai = sol.y[0, :]*ai_unit 

Li_x = sol.y[1, :]*Li_unit 
Li_y = sol.y[2, :]*Li_unit 
Li_z = sol.y[3, :]*Li_unit 
Li = np.sqrt(Li_x**2. + Li_y**2. + Li_z**2.)

ei_x = sol.y[4, :]
ei_y = sol.y[5, :]
ei_z = sol.y[6, :]
ei=np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)

S1_x = sol.y[7, :]*S1_unit
S1_y = sol.y[8, :]*S1_unit
S1_z = sol.y[9, :]*S1_unit
S1 = np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.)
S2_x = sol.y[10, :]*S2_unit 
S2_y = sol.y[11, :]*S2_unit
S2_z = sol.y[12, :]*S2_unit
S2 = np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.)


theta1_SL = np.real(np.arccos(\
        (S1_x*Li_x + S1_y*Li_y + S1_z*Li_z)/(S1*Li)+0j
                      ))
theta2_SL = np.real(np.arccos(\
        (S2_x*Li_x + S2_y*Li_y + S2_z*Li_z)/(S2*Li)+0j
                      ))
theta_SS = np.real(np.arccos(\
        (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)+0j
                    ))

## write data to an h5 file
fid = h5.File(data_dir + prefix + 'data.h5', 'w')
fid.attrs['M1'] = M1
fid.attrs['M2'] = M2
fid.attrs['chi1'] = chi1
fid.attrs['chi2'] = chi2
fid.attrs['br_flag'] = br_flag
fid.attrs['ss_flag'] = ss_flag

fid.create_dataset('t', shape=tt.shape, dtype=np.float, data=tt)
fid.create_dataset('ai', shape=ai.shape, dtype=np.float, data=ai)
fid.create_dataset('Li_x', shape=Li_x.shape, dtype=np.float, data=Li_x)
fid.create_dataset('Li_y', shape=Li_y.shape, dtype=np.float, data=Li_y)
fid.create_dataset('Li_z', shape=Li_z.shape, dtype=np.float, data=Li_z)
fid.create_dataset('ei_x', shape=ei_x.shape, dtype=np.float, data=ei_x)
fid.create_dataset('ei_y', shape=ei_y.shape, dtype=np.float, data=ei_y)
fid.create_dataset('ei_z', shape=ei_z.shape, dtype=np.float, data=ei_z)
fid.create_dataset('S1_x', shape=S1_x.shape, dtype=np.float, data=S1_x)
fid.create_dataset('S1_y', shape=S1_y.shape, dtype=np.float, data=S1_y)
fid.create_dataset('S1_z', shape=S1_z.shape, dtype=np.float, data=S1_z)
fid.create_dataset('S2_x', shape=S2_x.shape, dtype=np.float, data=S2_x)
fid.create_dataset('S2_y', shape=S2_y.shape, dtype=np.float, data=S2_y)
fid.create_dataset('S2_z', shape=S2_z.shape, dtype=np.float, data=S2_z)

fid.close()

## plot fig
fig=plt.figure(figsize=(9, 14))
ax=fig.add_subplot(411)
ax.semilogy(tt/P_yr, ai/AU, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$a_{\rm in}$ [AU]')

ax=fig.add_subplot(412)
ax.semilogy(tt/P_yr, 1.-ei, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$1-e_{\rm in}$')

ax=fig.add_subplot(413)
ax.plot(tt/P_yr, theta1_SL*180./np.pi, alpha=0.8, label=r'$S_1$', color='tab:grey')
ax.plot(tt/P_yr, theta2_SL*180./np.pi, alpha=0.8, label=r'$S_2$', color='tab:olive')
ax.axhline(90., color='tab:red', ls=':', alpha=0.5)
ax.set_ylabel(r'$\theta_{SL}$ [deg]')
ax.legend(loc='upper left')

ax=fig.add_subplot(414)
ax.plot(tt/P_yr, theta_SS*180./np.pi, alpha=0.8, color='tab:grey')
ax.set_ylabel(r'$\theta_{SS}$ [deg]')
ax.set_xlabel(r'Time [yr]')

plt.subplots_adjust(hspace=0)
plt.savefig(fig_dir+prefix+'evol.pdf')
plt.close()



