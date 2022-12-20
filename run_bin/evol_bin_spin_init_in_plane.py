#!/usr/bin/env python

import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import h5py as h5
import argparse, os, sys

sys.path.append('../')

from myConstants import *
import LKlib as LK

#######################################################################
### input 
#######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=np.int, default=0)
parser.add_argument('--Mt-Ms', type=np.float, default=50)
parser.add_argument('--qq', type=np.float, default=0.8)
parser.add_argument('--chi1', type=np.float, default=0.7)
parser.add_argument('--chi2', type=np.float, default=0.7)
parser.add_argument('--out-base', type=str, default='')
parser.add_argument('--out-dir', type=str, default='data/')

kwargs = parser.parse_args()
# convert it to a dict
kwargs=vars(kwargs)

idx = kwargs['idx']
out_base = kwargs['out_base']
out_dir = kwargs['out_dir']
Mt_Ms = kwargs['Mt_Ms']
qq = kwargs['qq']
chi1 = kwargs['chi1']
chi2 = kwargs['chi2']

# out_base += 'Mt_%.0f_q_%.2f_x1_%.2f_x2_%.2f_'%(Mt_Ms, qq, chi1, chi2)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
#######################################################################
### fixed pars
#######################################################################


M1 = Mt_Ms / (1.+qq) * Ms
M2 = M1 * qq
Mt = M1 + M2
mu = M1 * M2 / Mt
eta = mu / Mt

S1 = chi1 * G*M1**2./c
S2 = chi2 * G*M2**2./c

r_Mt = G*Mt/c**2.
t_Mt = r_Mt/c
t_Mt_pi = t_Mt * np.pi

S_Mt = G*Mt**2./c

a_unit = r_Mt
t_unit = 1e3*t_Mt
L_unit = S_Mt
S1_unit = S_Mt
S2_unit = S_Mt

par = np.array([M1, M2, 0, 0, 
                1e3*t_Mt, S_Mt, 0, r_Mt, S_Mt, S_Mt, 
                1, 1])

#######################################################################
### sample initial at 500 Mt
#######################################################################

a_init = 500.*r_Mt 
L_init = mu * np.sqrt(G*Mt*a_init)

L_v_init = np.array([0, 0, 1]) * L_init

# cS1L = np.random.uniform(low=-1, high=1)
# cS2L = np.random.uniform(low=-1, high=1)

cS1L = np.random.randn() * 0.05
cS2L = np.random.randn() * 0.05

phiS1 = np.random.uniform(low=0, high=2.*np.pi)
phiS2 = np.random.uniform(low=0, high=2.*np.pi)

sS1L = np.sqrt(1.-cS1L**2.)
sS2L = np.sqrt(1.-cS2L**2.)

S1_v_init = np.array([
    sS1L * np.cos(phiS1), 
    sS1L * np.sin(phiS1), 
    cS1L]) * S1
S2_v_init = np.array([
    sS2L * np.cos(phiS2), 
    sS2L * np.sin(phiS2), 
    cS2L]) * S2

a_nat_init = a_init / a_unit
L_nat_v_init = L_v_init / L_unit
e_v = np.zeros(3)
S1_nat_v_init = S1_v_init / S1_unit
S2_nat_v_init = S2_v_init / S2_unit

y_nat_init = np.hstack([
    a_nat_init, \
    L_nat_v_init, e_v, \
    S1_nat_v_init, S2_nat_v_init])

#######################################################################
### define events
#######################################################################

# functions to find specific separations
def find_a_Mt(t_nat, y_nat, par, 
           a_Mt_trgt=6):
    a_Mt = y_nat[0]
    resi = a_Mt - a_Mt_trgt
    return resi

event1=lambda t_nat_, y_nat_vect_:find_a_Mt(t_nat_, y_nat_vect_, par, a_Mt_trgt=50)
event1.direction = -1
event1.terminal = False

event2=lambda t_nat_, y_nat_vect_:find_a_Mt(t_nat_, y_nat_vect_, par, a_Mt_trgt=25)
event2.direction = -1
event2.terminal = False

event3=lambda t_nat_, y_nat_vect_:find_a_Mt(t_nat_, y_nat_vect_, par, a_Mt_trgt=10)
event3.direction = -1
event3.terminal = False

event4=lambda t_nat_, y_nat_vect_:find_a_Mt(t_nat_, y_nat_vect_, par, a_Mt_trgt=6)
event4.direction = -1
event4.terminal = True


#######################################################################
### integration
#######################################################################

int_func = lambda t_nat_, y_nat_vect_: LK.evol_binary(t_nat_, y_nat_vect_, par)

# t_run0 = timeit.default_timer()
sol = integ.solve_ivp(int_func, \
        t_span=(0, 1e9), y0=y_nat_init, rtol=1e-12, atol=1e-12, \
        events=[event1, event2, event3, event4])
# t_run1 = timeit.default_timer()

#######################################################################
### write outputs
#######################################################################

# write initial condition
out_file = out_dir + out_base + 'e0.h5'
fid = h5.File(out_file, 'a')
grp_id = '%i'%idx
if grp_id in fid.keys():
    del fid[grp_id]
grp = fid.create_group(grp_id)
grp.create_dataset('y_event', shape=y_nat_init.shape, dtype=y_nat_init.dtype, data=y_nat_init)
fid.close()

# write events
n_events = len(sol.y_events)
for i in range(n_events):
    y_event = sol.y_events[i]
    out_file = out_dir + out_base + 'e%i.h5'%(i+1)
    
    fid = h5.File(out_file, 'a')
    grp_id = '%i'%idx
    if grp_id in fid.keys():
        del fid[grp_id]
    grp = fid.create_group(grp_id)
    grp.create_dataset('y_event', shape=y_event.shape, dtype=y_event.dtype, data=y_event)
    fid.close()
    
