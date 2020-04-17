#!/usr/bin/env python

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d as g_filt
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
                     'legend.fontsize': 'large',
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
parser.add_argument('--t-m-cut', type=np.float, default=-1)
parser.add_argument('--chi-eff-cut', type=np.float, default=0.1)
parser.add_argument('--atol', type=np.float, default=3e-12)
parser.add_argument('--rtol', type=np.float, default=3e-12)
parser.add_argument('--nPt', type=np.int, default = 100)
parser.add_argument('--plot-flag', type=np.int, default=0)

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)

run_id=kwargs['run_id']
t_m_cut = kwargs['t_m_cut']
chi_eff_cut = kwargs['chi_eff_cut']
atol, rtol = kwargs['atol'], kwargs['rtol']
nPt = kwargs['nPt']
br_flag, ss_flag=1, 1
plot_flag = kwargs['plot_flag']

if chi_eff_cut<0.15:
    fig_dir = '/home/hang.yu/public_html/astro/LK_evol/LK2merger/fix_init_spin_ang/bin2merg/M3_1.0e+09Ms_ao_0.060pc_ai0_3.0AU/chi_eff_0/'
    data_dir = 'data/fix_init_spin_ang/bin2merg/M3_1.0e+09Ms_ao_0.060pc_ai0_3.0AU/chi_eff_0/'
else:
    fig_dir =\
    '/home/hang.yu/public_html/astro/LK_evol/LK2merger/fix_init_spin_ang/bin2merg/M3_1.0e+09Ms_ao_0.060pc_ai0_3.0AU/chi_eff_%.2f/'%chi_eff_cut
    data_dir = 'data/fix_init_spin_ang/bin2merg/M3_1.0e+09Ms_ao_0.060pc_ai0_3.0AU/chi_eff_%.2f/'%chi_eff_cut
    
prefix = 'id_%i_'%(run_id)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
### read in the data from a_i = 300 M_t as the initial condition ### 
data_300_dir = 'data/fix_init_spin_ang/DA/M3_1.0e+09Ms_ao_0.060pc_ai0_3.0AU/'
data_300 = np.zeros([0, 10])
data_LK = np.zeros([0, 15])
nFile=10

# note that the 300 r_Mt files starts from an id of 100!!!
for i in range(100, 100+nFile, 1):
    if os.path.exists(data_300_dir + 'id_%i_r_300_cond.txt'%i):
        data_300_ = np.loadtxt(data_300_dir + 'id_%i_r_300_cond.txt'%i)
        data_300 = np.vstack([data_300, data_300_])
        
        data_LK_ = np.loadtxt(data_300_dir + 'id_%i_LK_cond.txt'%i)
        data_LK = np.vstack([data_LK, data_LK_])
        
### apply cuts ###
chi_eff, t_m = data_LK[:, 4], data_LK[:, 8]
u_m_e = data_LK[:, 7]
idx = (np.abs(chi_eff)<chi_eff_cut) & (t_m>t_m_cut)

data_300 = data_300[idx, :]
t_m, u_m_e = t_m[idx], u_m_e[idx]
del data_LK

### draw one set of {J, L} ###
nSamp = data_300.shape[0]
samp_id = stats.randint.rvs(0, nSamp-1)
print('Number of samples aft cut, sample id:', nSamp, samp_id)
data_300 = data_300[samp_id, :]
t_m, u_m_e = t_m[samp_id], u_m_e[samp_id]
print('Merger time:%.3e, 1-e_max:%.3e'%(t_m, u_m_e))

M1, M2, chi1, chi2, chi_eff = data_300[:5]

M1, M2 = M1*Ms, M2*Ms
Mt = M1+M2
qq = M2/M1
mu = M1*M2/Mt
eta = mu/Mt

r_Mt = G*Mt/c**2.
S_Mt = G*Mt**2./c

S1, S2 = chi1*G*M1**2./c, chi2*G*M2**2./c

a_0 = 300.*r_Mt
J_0, L_0, e_0 = data_300[5:8]
J_0, L_0 = J_0*S_Mt, L_0*S_Mt

## draw S 
par_JL = np.array([M1, M2, S1, S2, chi_eff])
S_m, S_p = LK.find_Smp(J_0, L_0, e_0, par_JL, nPt=nPt)
S_v = np.linspace(S_m, S_p, nPt)
dSdt = np.zeros(nPt)
for i in range(nPt):
    dSdt[i] = LK.get_dSdt(J_0, L_0, e_0, S_v[i], par_JL)

pmf = 1./np.abs(dSdt)
# hack the boundaries using extrapolation
pmf[0]  = 2*pmf[1]-pmf[2] + 0.5*(pmf[3]+pmf[1]-2.*pmf[2]) 
pmf[-1] = 2*pmf[-2]-pmf[-3] + 0.5*(pmf[-4]+pmf[-2]-2.*pmf[-3]) 

pdf_S = pmf/integ.trapz(pmf, S_v/S_Mt)
# smooth the curve
pdf_S = g_filt(pdf_S, np.int(np.round(nPt/20)))
pdf_S = pdf_S/integ.trapz(pdf_S, S_v/S_Mt)

pdf_S_func = interp.interp1d(S_v/S_Mt, pdf_S, bounds_error=False, fill_value=0)

class S_pdf_class(stats.rv_continuous):
    def _pdf(self,SS):
        pdf = pdf_S_func(SS)
        return pdf
S_0 = S_pdf_class(a=S_m/S_Mt, b=S_p/S_Mt, xtol=1.e-10).rvs() * S_Mt
print('J_0/S_Mt, L_0/S_Mt, S_0/S_Mt, chi_eff', J_0/S_Mt, L_0/S_Mt, S_0/S_Mt, chi_eff)


#######################################################################
### Initial vectors
#######################################################################

phi_S1_0 = 0.
I_S1_0, I_S2_0, th_12_ref, phi_S2_0 \
    = LK.get_angles(J_0, L_0, e_0, S_0, par_JL)

chi1_x, chi1_y, chi1_z = \
    chi1*np.sin(I_S1_0)*np.cos(phi_S1_0), \
    chi1*np.sin(I_S1_0)*np.sin(phi_S1_0), \
    chi1*np.cos(I_S1_0)

chi2_x, chi2_y, chi2_z = \
    chi2*np.sin(I_S2_0)*np.cos(phi_S2_0), \
    chi2*np.sin(I_S2_0)*np.sin(phi_S2_0), \
    chi2*np.cos(I_S2_0)

uS1_0 = np.array([chi1_x, chi1_y, chi1_z])/chi1
uS2_0 = np.array([chi2_x, chi2_y, chi2_z])/chi2
uL_0 = np.array([0, 0, 1])

print('initial SS alignment', th_12_ref*180./np.pi)
print('ratio of thSS calced in two ways', np.inner(uS1_0, uS2_0)/np.cos(th_12_ref))

t_GW_0 = LK.get_inst_t_gw_from_a_orb(M1, M2, a_0, e_0)
print('T_gw/yr', '%e'%(t_GW_0/P_yr))
t_GW_isco = LK.get_inst_t_gw_from_a_orb(M1, M2, 6.*r_Mt, 0.)

t_unit=np.sqrt(t_GW_0*t_GW_isco)
L_unit = np.sqrt(L_0*S_Mt)
a_unit = np.sqrt(a_0*r_Mt)
S1_unit = S1
S2_unit = S2
par_LK = np.array([M1, M2, np.inf, np.inf, \
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

#######################################################################
### solve ode
#######################################################################

# to isco
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
    resi = (ai - 6*r_Mt)/ai_unit
    return resi

term_func=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par_LK)
term_func.direction = -1
term_func.terminal = True

int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_binary(t_nat_, y_nat_vect_, par_LK)
    
sol=integ.solve_ivp(int_func, \
        t_span=(0, 1e9), y0=y_nat_init, rtol=rtol, atol=atol, \
        events=term_func)

t_run1 = timeit.default_timer()
print('run time (to ISCO):', t_run1 - t_run0)


# to 4 M
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
    resi = (ai - 4*r_Mt)/ai_unit
    return resi

term_func2=lambda t_nat_, y_nat_vect_:terminator(t_nat_, y_nat_vect_, par_LK)
term_func2.direction = -1
term_func2.terminal = True

t2 = sol.t[-1]
y_nat_init2 = sol.y[:, -1]
sol2=integ.solve_ivp(int_func, \
        t_span=(t2, t2+1e9), y0=y_nat_init2, rtol=rtol, atol=atol, \
        events=term_func2)

t_run1 = timeit.default_timer()
print('run time (ISCO to 4 M):', t_run1 - t_run0)

#######################################################################
### output
#######################################################################

# get sol at ISCO
tt = sol.t*t_unit

a_orb = sol.y[0, :]*a_unit 
f_gw = np.sqrt(G*Mt/a_orb**3.)/np.pi

L_x = sol.y[1, :]*L_unit 
L_y = sol.y[2, :]*L_unit 
L_z = sol.y[3, :]*L_unit 
L_orb = np.sqrt(L_x**2. + L_y**2. + L_z**2.)

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

print('Final J, L, S:', J[-1]/S_Mt, L_orb[-1]/S_Mt, S[-1]/S_Mt)

theta1_SL = np.real(np.arccos(\
        (S1_x*L_x + S1_y*L_y + S1_z*L_z)/(S1*L_orb)+0j
                      ))
theta2_SL = np.real(np.arccos(\
        (S2_x*L_x + S2_y*L_y + S2_z*L_z)/(S2*L_orb)+0j
                      ))
theta_SS = np.real(np.arccos(\
        (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)+0j
                    ))

fid = open(data_dir + prefix + 'bin_evol.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
            J_0/S_Mt, L_0/S_Mt, S_0/S_Mt, \
            J[-1]/S_Mt, L_orb[-1]/S_Mt, S[-1]/S_Mt, \
            theta1_SL[-1], theta2_SL[-1], theta_SS[-1]))
fid.close()

fid = open(data_dir + prefix + 'bin_pre_cond.txt', 'a')
fid.write('%.9e\t%.6e\n'%(u_m_e, t_m))
fid.close()

if plot_flag:
    chi1_p, chi2_p = np.zeros(len(tt)), np.zeros(len(tt))
    chi_eff_num = np.zeros(len(tt))

    tau_gw_inst, tau_pre, tau_pre_inst \
        = np.zeros(len(tt)), np.zeros(len(tt)), np.zeros(len(tt))

    for i in range(len(tt)):
        chi1_p[i]=np.linalg.norm(LK.cross(\
                       np.array([S1_x[i], S1_y[i], S1_z[i]]), \
                       np.array([L_x[i],  L_y[i],  L_z[i]]))\
                     /L_orb[i]/(G*M1**2./c))
        chi2_p[i]=np.linalg.norm(LK.cross(\
                       np.array([S2_x[i], S2_y[i], S2_z[i]]), \
                       np.array([L_x[i],  L_y[i],  L_z[i]]))\
                     /L_orb[i]/(G*M2**2./c))
        chi_eff_num[i] = (M1*LK.inner(np.array([S1_x[i], S1_y[i], S1_z[i]]), \
                                      np.array([L_x[i],  L_y[i],  L_z[i] ]))\
                              /L_orb[i]/(G*M1**2./c)\
                     +M2*LK.inner(np.array([S2_x[i], S2_y[i], S2_z[i]]), \
                                  np.array([L_x[i],  L_y[i],  L_z[i] ]))\
                              /L_orb[i]/(G*M2**2./c))\
                    /(M1+M2)
    
        tau_gw_inst[i] = LK.get_inst_t_gw_from_a_orb(M1, M2, a_orb[i], 0)
        tau_pre[i] = LK.get_tau_pre(J[i], L_orb[i], 0, par_JL, nPt=100)
        tau_pre_inst[i] = S[i]/np.abs(LK.get_dSdt(J[i], L_orb[i], 0, S[i], par_JL))
        
    fig=plt.figure(figsize=(9, 14))
    ax=fig.add_subplot(211)
    ax.semilogx(f_gw, theta1_SL*180./np.pi, alpha=0.5, label=r'$\theta_{S_1 L}$')
    ax.semilogx(f_gw, theta2_SL*180./np.pi, alpha=0.5, label=r'$\theta_{S_2 L}$')
    ax.semilogx(f_gw, theta_SS*180./np.pi, ls=':', alpha=0.5, label=r'$\theta_{S_1 S_2}$')
    ax.legend(loc='lower left')
    ax.set_ylabel(r'$\theta$ [$^\circ$]')
#     ax.invert_xaxis()
    ax.set_xticklabels([])
    ax.set_ylim([0., 180.])
    
#     ax=fig.add_subplot(312)
#     ax.semilogx(f_gw, S/S_Mt)
#     ax.set_ylabel(r'$S/M_{\rm t}^2$')
# #     ax.invert_xaxis()
    
    ax=fig.add_subplot(212)
    
#     ax.semilogx((tt[-1]-tt), chi_eff_num, label=r'$\chi_{\rm eff}$')
#     ax.semilogx((tt[-1]-tt), chi1_p, label=r'$\chi_{1,\bot}$')
#     ax.semilogx((tt[-1]-tt), chi2_p, label=r'$\chi_{2,\bot}$')

    ax.loglog(f_gw, tau_gw_inst, label=r'$\tau_{\rm gw}$')
    ax.loglog(f_gw, tau_pre, label=r'$\tau_{SL}$', ls=':', alpha=0.5)
    ax.loglog(f_gw, tau_pre_inst, label=r'$\tau_{SL,\,{\rm ins}}$', ls='--', alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$\tau$ [yrs]')
    ax.set_xlabel(r'$f_{\rm gw}$ [Hz]')
    
    plt.subplots_adjust(hspace=0)
    plt.savefig(fig_dir + prefix + 'spin_bin.pdf')
    plt.close()
    

    
# get sol at 4 M 
tt = sol2.t*t_unit

a_orb = sol2.y[0, :]*a_unit 
f_gw = np.sqrt(G*Mt/a_orb**3.)/np.pi

L_x = sol2.y[1, :]*L_unit 
L_y = sol2.y[2, :]*L_unit 
L_z = sol2.y[3, :]*L_unit 
L_orb = np.sqrt(L_x**2. + L_y**2. + L_z**2.)

S1_x = sol2.y[7, :]*S1_unit
S1_y = sol2.y[8, :]*S1_unit
S1_z = sol2.y[9, :]*S1_unit
S1 = np.median(np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.))

S2_x = sol2.y[10, :]*S2_unit 
S2_y = sol2.y[11, :]*S2_unit
S2_z = sol2.y[12, :]*S2_unit
S2 = np.median(np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.))

S_x = S1_x + S2_x
S_y = S1_y + S2_y
S_z = S1_z + S2_z
S = np.sqrt(S_x**2. + S_y**2. + S_z**2.)

J_x = L_x + S1_x + S2_x
J_y = L_y + S1_y + S2_y
J_z = L_z + S1_z + S2_z
J = np.sqrt(J_x**2. + J_y**2. + J_z**2.)

print('4 Mt J, L, S:', J[-1]/S_Mt, L_orb[-1]/S_Mt, S[-1]/S_Mt)

theta1_SL = np.real(np.arccos(\
        (S1_x*L_x + S1_y*L_y + S1_z*L_z)/(S1*L_orb)+0j
                      ))
theta2_SL = np.real(np.arccos(\
        (S2_x*L_x + S2_y*L_y + S2_z*L_z)/(S2*L_orb)+0j
                      ))
theta_SS = np.real(np.arccos(\
        (S1_x*S2_x + S1_y*S2_y + S1_z*S2_z)/(S1*S2)+0j
                    ))

fid = open(data_dir + prefix + 'bin_evol_4Mt.txt', 'a')
fid.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'\
          %(M1/Ms, M2/Ms, chi1, chi2, chi_eff, \
            J_0/S_Mt, L_0/S_Mt, S_0/S_Mt, \
            J[-1]/S_Mt, L_orb[-1]/S_Mt, S[-1]/S_Mt, \
            theta1_SL[-1], theta2_SL[-1], theta_SS[-1]))
fid.close()