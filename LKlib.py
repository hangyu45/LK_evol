import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import h5py as h5
from numba import jit, prange

from myConstants import *

@jit(nopython=True)
def inner(xx, yy):
    return np.sum(xx*yy)

@jit(nopython=True)
def cross(xx, yy):
    zz=np.array([\
         xx[1]*yy[2] - xx[2]*yy[1], \
         xx[2]*yy[0] - xx[0]*yy[2], \
         xx[0]*yy[1] - xx[1]*yy[0]
                ])
    return zz

@jit(nopython=True, fastmath=True)
def get_inst_t_gw_from_a_orb(M1, M2, a_orb, e):
    Mt=M1+M2
    mu=M1*M2/(M1+M2)
    inv_t_gw = (64./5.)*(G**3./c**5.)*mu*Mt**2./a_orb**4. \
                * (1.+73./24.*e**2+37./96.*e**4.)/(1.-e**2.)**(3.5)
    t_gw = 1./inv_t_gw
    return t_gw

@jit(nopython=True, fastmath=True)
def get_t_lk(M1, M2, M3, ai, ao):
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    t_lk = 1./omega_i * (M1+M2)/M3 * (ao/ai)**3.
    return t_lk


@jit(nopython=True, fastmath=True)
def get_dy_orb_GR_GW(y_orb_vect, par, par_GR):
    """
    GR precession & GW decay
    """
    # parse input
    ai, \
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z\
                = y_orb_vect
    
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
        
    # par for GR & GW calc
    mu_i, omega_i,\
    Li_e0, ei, eff_i,\
    uLi_x, uLi_y, uLi_z\
                = par_GR
        
    # scalars
    omega_GR = 3.*G*(M1+M2)/(c**2.*ai*eff_i**2.) * omega_i
    
    G3_c5 = G**3./c**5.
    ei2 = ei**2.
    ei4 = ei**4.
    dai = - (64./5.*G3_c5) * (mu_i*(M1+M2)**2./(ai**3.))\
          * (1. + 73./24.*ei2 + 37./96.*ei4)/(eff_i**7.)    
    dLi_GW = - (32./5.*G3_c5*np.sqrt(G)) * mu_i**2.*(M1+M2)**2.5 / (ai**3.5) \
          * (1.+0.875*ei2)/(eff_i**4.)
    dei_GW = - (304./15.*G3_c5) * mu_i*(M1+M2)**2./(ai**4.)\
          * (1.+121./304.*ei2)/(eff_i**5.)
        
    # vectors
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x, ei_y, ei_z])
    uLi_c_ei = cross(uLi_v, ei_v)
    
    # GR precession 
    dei_GR_v = omega_GR * uLi_c_ei
    
    # GW radiation
    dLi_GW_v = dLi_GW * uLi_v
    dei_GW_v = dei_GW * ei_v
        
    # total
    dai =np.array([dai])
    dLi_v = dLi_GW_v
    dei_v = dei_GR_v + dei_GW_v
    
    dy_orb_vect = np.array([\
                    dai[0], \
                    dLi_v[0], dLi_v[1], dLi_v[2], \
                    dei_v[0], dei_v[1], dei_v[2]])
    return dy_orb_vect


@jit(nopython=True, fastmath=True)
def get_dy_LK_quad_da(y_LK_vect, par, par_LK):
    """
    Lidov-Kozai
    """
    # parse input
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
    Lo_x, Lo_y, Lo_z, eo_x, eo_y, eo_z\
                = y_LK_vect
        
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
        
    # par for LK calc
    mu_i, mu_o, omega_i, ai, \
    Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
    uLi_x, uLi_y, uLi_z, \
    uLo_x, uLo_y, uLo_z\
                = par_LK

    # scalar quantities
    #dLi_e0 = 0.5 * Li_e0 / ai * dai # the GW loss has been accounted for
    t_LK = (1./omega_i) * (M1+M2)/M3 * (ao*eff_o/ai)**3.
    p75_t_LK = 0.75/t_LK
    
    # directional products
    ji_v  = np.array([uLi_x, uLi_y, uLi_z]) * eff_i
    ei_v  = np.array([ei_x,  ei_y,  ei_z]) 
    uLo_v = np.array([uLo_x, uLo_y, uLo_z])
    
    ji_d_uLo = inner(ji_v, uLo_v)
    ei_d_uLo = inner(ei_v, uLo_v)
    
    ji_c_uLo_v = cross(ji_v, uLo_v)
    ei_c_uLo_v = cross(ei_v, uLo_v)
    ji_c_ei_v  = cross(ji_v, ei_v)
    
    # derivatives of dir vects 
    dji_v = p75_t_LK * (ji_d_uLo*ji_c_uLo_v - 5.*ei_d_uLo*ei_c_uLo_v)
    dei_v = p75_t_LK * (ji_d_uLo*ei_c_uLo_v + 2.*ji_c_ei_v - 5.*ei_d_uLo*ji_c_uLo_v)
    djo_v = p75_t_LK * Li_e0/Lo_e0 * (-ji_d_uLo*ji_c_uLo_v + 5.*ei_d_uLo*ei_c_uLo_v)
    deo_v = np.zeros(3) # FIXME; currently only consider the case that eo stays zero
    
    # derivatives on the angular momenta
    # the GW part have been accounted for in get_dy_orb_GR_GW()
    # should not need to include it again in LK
    dLi_v = Li_e0 * dji_v
    dLo_v = Lo_e0 * djo_v
    
    dy_LK_vect = np.array([\
                 dLi_v[0], dLi_v[1], dLi_v[2], \
                 dei_v[0], dei_v[1], dei_v[2], \
                 dLo_v[0], dLo_v[1], dLo_v[2], \
                 deo_v[0], deo_v[1], deo_v[2]\
        ])
    
    return dy_LK_vect

@jit(nopython=True, fastmath=True)
def get_dy_SP(y_SP_vect, par, par_SP):
    """
    de Sitter spin-orbit & Lense Thirring spin-spin
    """
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z, \
    S1_x, S1_y, S1_z, \
    S2_x, S2_y, S2_z \
        = y_SP_vect
        
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
        
    # par for LK calc
    mu_i, omega_i, ai, \
    Li_e0, ei, eff_i, S1, S2, \
    uLi_x, uLi_y, uLi_z, \
    uS1_x, uS1_y, uS1_z, uS2_x, uS2_y, uS2_z\
                = par_SP
        
    # scalar quantities
    G_c2 = G/c**2.
    ai3_eff_i3 = (ai*eff_i)**3.
    omega1_SL = 1.5*G_c2*(M2+mu_i/3.)/(ai*eff_i**2.)*omega_i
    omega2_SL = 1.5*G_c2*(M1+mu_i/3.)/(ai*eff_i**2.)*omega_i
    omega1_SL_br = 0.5*G_c2*S1*(4.+3.*M2/M1)/(ai3_eff_i3)
    omega2_SL_br = 0.5*G_c2*S2*(4.+3.*M1/M2)/(ai3_eff_i3)
    
    omega1_SL_S1 = omega1_SL * S1
    omega2_SL_S2 = omega2_SL * S2
    omega1_SL_br_Li = omega1_SL_br*Li_e0*eff_i * br_flag
    omega2_SL_br_Li = omega2_SL_br*Li_e0*eff_i * br_flag
    
    omega1_SS = 0.5*G_c2*S2/(ai3_eff_i3)
    omega2_SS = 0.5*G_c2*S1/(ai3_eff_i3)
    omega_SS_br = -1.5*G_c2*S1*S2/(mu_i*omega_i)/(ai3_eff_i3*ai**2.*eff_i)
    
    omega1_SS_S1 = omega1_SS * S1 * ss_flag
    omega2_SS_S2 = omega2_SS * S2 * ss_flag
    omega_SS_br_Li = omega_SS_br*Li_e0*eff_i * ss_flag * br_flag
    
    # directional products 
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x,  ei_y,  ei_z ])
    uS1_v = np.array([uS1_x, uS1_y, uS1_z])
    uS2_v = np.array([uS2_x, uS2_y, uS2_z])
    
    uLi_d_uS1 = inner(uLi_v, uS1_v)
    uLi_d_uS2 = inner(uLi_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uLi_c_uS1_v = cross(uLi_v, uS1_v)
    uLi_c_uS2_v = cross(uLi_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    uS1_c_ei_v  = cross(uS1_v, ei_v )
    uS2_c_ei_v  = cross(uS2_v, ei_v )
    uLi_c_ei_v  = cross(uLi_v, ei_v ) 
    
    # de Sitter
    dLi_SL_v = omega1_SL_br_Li * (-uLi_c_uS1_v)\
             + omega2_SL_br_Li * (-uLi_c_uS2_v)
    dei_SL_v = omega1_SL_br * (uS1_c_ei_v - 3.*uLi_d_uS1*uLi_c_ei_v)\
             + omega2_SL_br * (uS2_c_ei_v - 3.*uLi_d_uS2*uLi_c_ei_v)
    dS1_SL_v = omega1_SL_S1 * (uLi_c_uS1_v)
    dS2_SL_v = omega2_SL_S2 * (uLi_c_uS2_v)
    
    # Lense-Thirring
    dLi_SS_v = omega_SS_br_Li * (- uLi_d_uS1*uLi_c_uS2_v - uLi_d_uS2*uLi_c_uS2_v)
    dei_SS_v = omega_SS_br * (uLi_d_uS1*uS2_c_ei_v + uLi_d_uS2*uS2_c_ei_v\
                             +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_v)
    dS1_SS_v = omega1_SS_S1 * (- uS1_c_uS2_v - 3.*uLi_d_uS2*uLi_c_uS1_v)
    dS2_SS_v = omega2_SS_S2 * (+ uS1_c_uS2_v - 3.*uLi_d_uS1*uLi_c_uS2_v)
    
    # total 
    dLi_v = dLi_SL_v + dLi_SS_v
    dei_v = dei_SL_v + dei_SS_v
    dS1_v = dS1_SL_v + dS1_SS_v
    dS2_v = dS2_SL_v + dS2_SS_v
    
    dy_SP_vect = np.array([\
                 dLi_v[0], dLi_v[1], dLi_v[2], \
                 dei_v[0], dei_v[1], dei_v[2], \
                 dS1_v[0], dS1_v[1], dS1_v[2], \
                 dS2_v[0], dS2_v[1], dS2_v[2]])
    return dy_SP_vect

@jit(nopython=True, fastmath=True, parallel=True)
def evol_LK_quad_da(t_nat, y_nat_vect, par):
    # parse parameters
    # 0
    # 1-6
    # 7-12
    # 13-15
    # 16-18
    ai_nat, \
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
    
    # convert to cgs units
    ai = ai_nat * ai_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    Lo_v = np.array([Lo_nat_x, Lo_nat_y, Lo_nat_z]) * Lo_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    S2_v = np.array([S2_nat_x, S2_nat_y, S2_nat_z]) * S2_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    Li_e0 = mu_i*np.sqrt(G*(M1+M2)*ai)
    Lo_e0 = mu_o*np.sqrt(G*(M1+M2+M3)*ao)
    
    ei = np.sqrt(inner(ei_v, ei_v))
    eo = np.sqrt(inner(eo_v, eo_v))
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    S1 = np.sqrt(inner(S1_v, S1_v))
    S2 = np.sqrt(inner(S2_v, S2_v))
    
    # unity vectors
    uLi_v = Li_v / (Li_e0 * eff_i)
    uLo_v = Lo_v / (Lo_e0 * eff_o)
    uS1_v = S1_v / S1
    uS2_v = S2_v / S2
    
    # get GR & GW terms
    y_orb_vect = np.array([ai, \
                           Li_v[0], Li_v[1], Li_v[2], \
                           ei_v[0], ei_v[1], ei_v[2]])
    par_GR = np.array([mu_i, omega_i,\
                       Li_e0, ei, eff_i,\
                       uLi_v[0], uLi_v[1], uLi_v[2]])

    dai,\
    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z\
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        
    # get LK terms
    y_LK_vect = np.array([Li_v[0], Li_v[1], Li_v[2], \
                          ei_v[0], ei_v[1], ei_v[2], \
                          Lo_v[0], Lo_v[1], Lo_v[2], \
                          eo_v[0], eo_v[1], eo_v[2]])
    par_LK = np.array([mu_i, mu_o, omega_i, ai, \
                        Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
                        uLi_v[0], uLi_v[1], uLi_v[2], \
                        uLo_v[0], uLo_v[1], uLo_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, \
    deo_LK_x, deo_LK_y, deo_LK_z\
        = get_dy_LK_quad_da(y_LK_vect, par, par_LK)
        
    # get SL & SS terms
    y_SP_vect = np.array([Li_v[0], Li_v[1], Li_v[2], \
                          ei_v[0], ei_v[1], ei_v[2], \
                          S1_v[0], S1_v[1], S1_v[2], \
                          S2_v[0], S2_v[1], S2_v[2]])
    par_SP = np.array([mu_i, omega_i, ai, \
                        Li_e0, ei, eff_i, S1, S2, \
                        uLi_v[0], uLi_v[1], uLi_v[2], \
                        uS1_v[0], uS1_v[1], uS1_v[2], \
                        uS2_v[0], uS2_v[1], uS2_v[2]])
    
    dLi_SP_x, dLi_SP_y, dLi_SP_z, \
    dei_SP_x, dei_SP_y, dei_SP_z, \
    dS1_SP_x, dS1_SP_y, dS1_SP_z, \
    dS2_SP_x, dS2_SP_y, dS2_SP_z\
        = get_dy_SP(y_SP_vect, par, par_SP)
            
    # total 
    # GW of semi-major axis
    dai_nat = dai / ai_unit
    
    # inner orb sees GR&GW + LK + SP back reaction
    dLi_nat_x = (dLi_GR_x + dLi_LK_x + dLi_SP_x) / Li_unit
    dLi_nat_y = (dLi_GR_y + dLi_LK_y + dLi_SP_y) / Li_unit
    dLi_nat_z = (dLi_GR_z + dLi_LK_z + dLi_SP_z) / Li_unit
    dei_x = dei_GR_x + dei_LK_x + dei_SP_x
    dei_y = dei_GR_y + dei_LK_y + dei_SP_y
    dei_z = dei_GR_z + dei_LK_z + dei_SP_z
    
    # outer orb sees only LK
    dLo_nat_x = dLo_LK_x / Lo_unit
    dLo_nat_y = dLo_LK_y / Lo_unit
    dLo_nat_z = dLo_LK_z / Lo_unit
    deo_x = deo_LK_x 
    deo_y = deo_LK_y 
    deo_z = deo_LK_z 
    
    # S1/S2 sees SP (de Sitter & Lense-Thirring)
    dS1_nat_x = dS1_SP_x / S1_unit
    dS1_nat_y = dS1_SP_y / S1_unit
    dS1_nat_z = dS1_SP_z / S1_unit
    dS2_nat_x = dS2_SP_x / S2_unit
    dS2_nat_y = dS2_SP_y / S2_unit
    dS2_nat_z = dS2_SP_z / S2_unit
    
    dy_nat_vect = np.array([\
            dai_nat, \
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, \
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, \
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
    return dy_nat_vect


