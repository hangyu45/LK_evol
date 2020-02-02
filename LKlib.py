import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import h5py as h5
from numba import jit, prange

from myConstants import *

def get_inst_t_gw_from_a_orb(M1, M2, a_orb, e):
    Mt=M1+M2
    mu=M1*M2/(M1+M2)
    inv_t_gw = (64./5.)*(G**3./c**5.)*mu*Mt**2./a_orb**4. \
                * (1.+73./24.*e**2+37./96.*e**4.)/(1.-e**2.)**(3.5)
    t_gw = 1./inv_t_gw
    return t_gw


def get_t_lk(M1, M2, M3, ai, ao):
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    t_lk = 1./omega_i * (M1+M2)/M3 * (ao/ai)**3.
    return t_lk
    

@jit(nopython=True, fastmath=True)
def get_dy_orb_GR_GW(y_orb_vect, par, par_GR):
    """
    GR precession & GW decay
    """
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, ai\
                = y_orb_vect
        
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    M1, M2, M3, ao,\
    br_flag, ss_flag\
                = par
        
    mu_i, omega_i,\
    Li_e0, ei, eff_i,\
    uLi_x, uLi_y, uLi_z\
                = par_GR
    
    # GR precession 
    omega_GR = 3.*G*(M1+M2)/(c**2.*ai*eff_i**2.) * omega_i
    
    uLi_c_ei_x = uLi_y*ei_z - uLi_z*ei_y
    uLi_c_ei_y = uLi_z*ei_x - uLi_x*ei_z
    uLi_c_ei_z = uLi_x*ei_y - uLi_y*ei_x
    
    dei_GR_x = omega_GR * uLi_c_ei_x
    dei_GR_y = omega_GR * uLi_c_ei_y
    dei_GR_z = omega_GR * uLi_c_ei_z
    
    # GW radiation
    G3_c5 = G**3./c**5.
    ei2 = ei**2.
    ei4 = ei**4.
    
    dai = - (64./5.*G3_c5) * (mu_i*(M1+M2)**2./(ai**3.))\
          * (1. + 73./24.*ei2 + 37./96.*ei4)/(eff_i**7.)
        
    dLi_GW = - (32./5.*G3_c5*np.sqrt(G)) * mu_i**2.*(M1+M2)**2.5 / (ai**3.5) \
          * (1.+0.875*ei2)/(eff_i**4.)
        
    dei_GW = - (304./15.*G3_c5) * mu_i*(M1+M2)**2./(ai**4.)\
          * (1.+121./304.*ei2)/(eff_i**5.)
        
    # total
    dLi_x = dLi_GW * uLi_x
    dLi_y = dLi_GW * uLi_y
    dLi_z = dLi_GW * uLi_z
    
    dei_x = dei_GR_x + dei_GW * ei_x
    dei_y = dei_GR_y + dei_GW * ei_y
    dei_z = dei_GR_z + dei_GW * ei_z
    
    dy_orb_vect = np.array([\
            dLi_x, dLi_y, dLi_z, dei_x, dei_y, dei_z, dai])
    return dy_orb_vect

@jit(nopython=True, fastmath=True)
def get_dy_LK_quad_da(y_LK_vect, par, par_LK):
    """
    Lidov-Kozai
    """
    # parse input
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
    Lo_x, Lo_y, Lo_z, eo_x, eo_y, eo_z, \
    ai\
                = y_LK_vect
        
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit,\
    M1, M2, M3, ao,\
    br_flag, ss_flag\
                = par
        
    mu_i, mu_o, omega_i, \
    Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
    ji_x, ji_y, ji_z,\
    jo_x, jo_y, jo_z, uLo_x, uLo_y, uLo_z,\
    dai \
                = par_LK

    # scalar quantities
    dLi_e0 = 0.5 * Li_e0 / ai * dai
    t_LK = (1./omega_i) * (M1+M2)/M3 * (ao*eff_o/ai)**3.
    p75_t_LK = 0.75/t_LK
    
    # directional products    
    ji_d_uLo = ji_x*uLo_x + ji_y*uLo_y + ji_z*uLo_z
    ei_d_uLo = ei_x*uLo_x + ei_y*uLo_y + ei_z*uLo_z
    
    ji_c_uLo_x = ji_y*uLo_z - ji_z*uLo_y
    ji_c_uLo_y = ji_z*uLo_x - ji_x*uLo_z
    ji_c_uLo_z = ji_x*uLo_y - ji_y*uLo_x
    ei_c_uLo_x = ei_y*uLo_z - ei_z*uLo_y
    ei_c_uLo_y = ei_z*uLo_x - ei_x*uLo_z
    ei_c_uLo_z = ei_x*uLo_y - ei_y*uLo_x
    ji_c_ei_x = ji_y*ei_z - ji_z*ei_y
    ji_c_ei_y = ji_z*ei_x - ji_x*ei_z
    ji_c_ei_z = ji_x*ei_y - ji_y*ei_x
    
    # derivatives of dir vects 
    dji_x = p75_t_LK * (ji_d_uLo*ji_c_uLo_x - 5.*ei_d_uLo*ei_c_uLo_x)
    dji_y = p75_t_LK * (ji_d_uLo*ji_c_uLo_y - 5.*ei_d_uLo*ei_c_uLo_y)
    dji_z = p75_t_LK * (ji_d_uLo*ji_c_uLo_z - 5.*ei_d_uLo*ei_c_uLo_z)
    
    dei_x = p75_t_LK * (ji_d_uLo*ei_c_uLo_x + 2.*ji_c_ei_x - 5.*ei_d_uLo*ji_c_uLo_x)
    dei_y = p75_t_LK * (ji_d_uLo*ei_c_uLo_y + 2.*ji_c_ei_y - 5.*ei_d_uLo*ji_c_uLo_y)
    dei_z = p75_t_LK * (ji_d_uLo*ei_c_uLo_z + 2.*ji_c_ei_z - 5.*ei_d_uLo*ji_c_uLo_z)
    
    djo_x = p75_t_LK * Li_e0/Lo_e0 * (-ji_d_uLo*ji_c_uLo_x + 5.*ei_d_uLo*ei_c_uLo_x) 
    djo_y = p75_t_LK * Li_e0/Lo_e0 * (-ji_d_uLo*ji_c_uLo_y + 5.*ei_d_uLo*ei_c_uLo_y)
    djo_z = p75_t_LK * Li_e0/Lo_e0 * (-ji_d_uLo*ji_c_uLo_z + 5.*ei_d_uLo*ei_c_uLo_z)
    
    # FIXME; currently only consider the case that eo stays zero
    deo_x = 0.
    deo_y = 0.
    deo_z = 0.
    
    # derivatives on the angular momenta
    dLi_x = Li_e0 * dji_x + dLi_e0 * ji_x
    dLi_y = Li_e0 * dji_y + dLi_e0 * ji_y
    dLi_z = Li_e0 * dji_z + dLi_e0 * ji_z
    
    dLo_x = Lo_e0 * djo_x
    dLo_y = Lo_e0 * djo_y
    dLo_z = Lo_e0 * djo_z
    
    dy_LK_vect = np.array([\
                    dLi_x, dLi_y, dLi_z, dei_x, dei_y, dei_z, \
                    dLo_x, dLo_y, dLo_z, deo_x, deo_y, deo_z, \
                    dai])
    return dy_LK_vect


@jit(nopython=True, fastmath=True)
def get_dy_SP(y_SP_vect, par, par_SP):
    """
    de Sitter spin-orbit & Lense Thirring spin-spin
    """
    
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z,\
    ai,\
    S1_x, S1_y, S1_z,\
    S2_x, S2_y, S2_z\
        = y_SP_vect
        
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    M1, M2, M3, ao, \
    br_flag, ss_flag\
                = par
        
    mu_i, omega_i, Li_e0, ei, eff_i, dai, \
    uLi_x, uLi_y, uLi_z, \
    S1, S2, \
    uS1_x, uS1_y, uS1_z, uS2_x, uS2_y, uS2_z\
                = par_SP
        
    # scalar quantities
    G_c2 = G/c**2.
    ai3_eff_i3 = (ai*eff_i)**3.
    omega1_SL = 1.5*G_c2*(M2+mu_i/3.)/(ai*eff_i**2.)*omega_i
    omega2_SL = 1.5*G_c2*(M1+mu_i/3.)/(ai*eff_i**2.)*omega_i
    omega1_SL_br = 0.5*G_c2*S1*(4.+3.*M2/M1)/(ai3_eff_i3)
    omega2_SL_br = 0.5*G_c2*S2*(4.+3.*M1/M2)/(ai3_eff_i3)
    #print('omega_SL:', omega1_SL, omega2_SL, omega1_SL_br, omega2_SL_br)
    
    omega1_SL_S1 = omega1_SL * S1
    omega2_SL_S2 = omega2_SL * S2
    omega1_SL_br_Li = omega1_SL_br*Li_e0*eff_i * br_flag
    omega2_SL_br_Li = omega2_SL_br*Li_e0*eff_i * br_flag
    #print('omega_SL * S', omega1_SL_S1, omega2_SL_S2, omega1_SL_br_Li, omega2_SL_br_Li)
    
    omega1_SS = 0.5*G_c2*S2/(ai3_eff_i3)
    omega2_SS = 0.5*G_c2*S1/(ai3_eff_i3)
    omega_SS_br = -1.5*G_c2*S1*S2/(mu_i*omega_i)/(ai3_eff_i3*ai**2.*eff_i)
    #print('omega_SS', omega1_SS, omega2_SS, omega_SS_br)
    
    omega1_SS_S1 = omega1_SS * S1 * ss_flag
    omega2_SS_S2 = omega2_SS * S2 * ss_flag
    omega_SS_br_Li = omega_SS_br*Li_e0*eff_i * ss_flag * br_flag
    #print('omega_SS * S', omega1_SS_S1, omega2_SS_S2, omega_SS_br_Li)
    
    # directional products 
    uLi_d_uS1 = (uLi_x*uS1_x + uLi_y*uS1_y + uLi_z*uS1_z)
    uLi_d_uS2 = (uLi_x*uS2_x + uLi_y*uS2_y + uLi_z*uS2_z)
    uS1_d_uS2 = (uS1_x*uS2_x + uS1_y*uS2_y + uS1_z*uS2_z)
    
    uLi_c_uS1_x = uLi_y*uS1_z - uLi_z*uS1_y
    uLi_c_uS1_y = uLi_z*uS1_x - uLi_x*uS1_z
    uLi_c_uS1_z = uLi_x*uS1_y - uLi_y*uS1_x
    uLi_c_uS2_x = uLi_y*uS2_z - uLi_z*uS2_y
    uLi_c_uS2_y = uLi_z*uS2_x - uLi_x*uS2_z
    uLi_c_uS2_z = uLi_x*uS2_y - uLi_y*uS2_x
    
    uS1_c_ei_x  = uS1_y* ei_z - uS1_z* ei_y
    uS1_c_ei_y  = uS1_z* ei_x - uS1_x* ei_z
    uS1_c_ei_z  = uS1_x* ei_y - uS1_y* ei_x
    uS2_c_ei_x  = uS2_y* ei_z - uS2_z* ei_y
    uS2_c_ei_y  = uS2_z* ei_x - uS2_x* ei_z
    uS2_c_ei_z  = uS2_x* ei_y - uS2_y* ei_x
    
    uLi_c_ei_x  = uLi_y* ei_z - uLi_z* ei_y
    uLi_c_ei_y  = uLi_z* ei_x - uLi_x* ei_z
    uLi_c_ei_z  = uLi_x* ei_y - uLi_y* ei_x
    
    uS1_c_uS2_x = uS1_y*uS2_z - uS1_z*uS2_y
    uS1_c_uS2_y = uS1_z*uS2_x - uS1_x*uS2_z
    uS1_c_uS2_z = uS1_x*uS2_y - uS1_y*uS2_x
    
    # spin back-reaction on the orbit
    dLi_x = omega1_SL_br_Li * (- uLi_c_uS1_x)\
          + omega2_SL_br_Li * (- uLi_c_uS2_x)\
          + omega_SS_br_Li  * (- uLi_d_uS1*uLi_c_uS2_x - uLi_d_uS2*uLi_c_uS1_x) 
    
    dLi_y = omega1_SL_br_Li * (- uLi_c_uS1_y)\
          + omega2_SL_br_Li * (- uLi_c_uS2_y)\
          + omega_SS_br_Li  * (- uLi_d_uS1*uLi_c_uS2_y - uLi_d_uS2*uLi_c_uS1_y) 
            
    dLi_z = omega1_SL_br_Li * (- uLi_c_uS1_z)\
          + omega2_SL_br_Li * (- uLi_c_uS2_z)\
          + omega_SS_br_Li  * (- uLi_d_uS1*uLi_c_uS2_z - uLi_d_uS2*uLi_c_uS1_z) 
            
    dei_x = omega1_SL_br * (uS1_c_ei_x - 3.*uLi_d_uS1*uLi_c_ei_x)\
          + omega2_SL_br * (uS2_c_ei_x - 3.*uLi_d_uS2*uLi_c_ei_x)\
          + omega_SS_br  * (uLi_d_uS1*uS2_c_ei_x + uLi_d_uS2*uS1_c_ei_x + \
                             +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_x)
            
    dei_y = omega1_SL_br * (uS1_c_ei_y - 3.*uLi_d_uS1*uLi_c_ei_y)\
          + omega2_SL_br * (uS2_c_ei_y - 3.*uLi_d_uS2*uLi_c_ei_y)\
          + omega_SS_br  * (uLi_d_uS1*uS2_c_ei_y + uLi_d_uS2*uS1_c_ei_y + \
                             +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_y)
           
    dei_z = omega1_SL_br * (uS1_c_ei_z - 3.*uLi_d_uS1*uLi_c_ei_z)\
          + omega2_SL_br * (uS2_c_ei_z - 3.*uLi_d_uS2*uLi_c_ei_z)\
          + omega_SS_br  * (uLi_d_uS1*uS2_c_ei_z + uLi_d_uS2*uS1_c_ei_z + \
                             +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_z)
            
    dS1_x = omega1_SL_S1 * (uLi_c_uS1_x)\
          + omega1_SS_S1 * (- uS1_c_uS2_x - 3.*uLi_d_uS2*uLi_c_uS1_x)
        
    dS1_y = omega1_SL_S1 * (uLi_c_uS1_y)\
          + omega1_SS_S1 * (- uS1_c_uS2_y - 3.*uLi_d_uS2*uLi_c_uS1_y)
        
    dS1_z = omega1_SL_S1 * (uLi_c_uS1_z)\
          + omega1_SS_S1 * (- uS1_c_uS2_z - 3.*uLi_d_uS2*uLi_c_uS1_z)
        
    dS2_x = omega2_SL_S2 * (uLi_c_uS2_x)\
          + omega2_SS_S2 * (uS1_c_uS2_x - 3.*uLi_d_uS1*uLi_c_uS2_x)
        
    dS2_y = omega2_SL_S2 * (uLi_c_uS2_y)\
          + omega2_SS_S2 * (uS1_c_uS2_y - 3.*uLi_d_uS1*uLi_c_uS2_y)
        
    dS2_z = omega2_SL_S2 * (uLi_c_uS2_z)\
          + omega2_SS_S2 * (uS1_c_uS2_z - 3.*uLi_d_uS1*uLi_c_uS2_z)
        
    dy_SP_vect = np.array([\
                dLi_x, dLi_y, dLi_z, dei_x, dei_y, dei_z, \
                dai, \
                dS1_x, dS1_y, dS1_z, \
                dS2_x, dS2_y, dS2_z\
                          ])
    return dy_SP_vect

@jit(nopython=True, fastmath=True)
def evol_LK_quad_da(t_nat, y_nat_vect, par):
    # parse parameters
    # 0-5
    # 6-11
    # 12
    # 13-15
    # 16-18
    
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    ai_nat, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    M1, M2, M3, ao,\
    br_flag, ss_flag\
                = par
    
    # convert to cgs units
    Li_x, Li_y, Li_z = Li_nat_x*Li_unit, Li_nat_y*Li_unit, Li_nat_z*Li_unit
    Lo_x, Lo_y, Lo_z = Lo_nat_x*Lo_unit, Lo_nat_y*Lo_unit, Lo_nat_z*Lo_unit
    ai = ai_nat * ai_unit
    
    S1_x, S1_y, S1_z = S1_nat_x*S1_unit, S1_nat_y*S1_unit, S1_nat_z*S1_unit
    S2_x, S2_y, S2_z = S2_nat_x*S2_unit, S2_nat_y*S2_unit, S2_nat_z*S2_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    Li_e0 = mu_i*np.sqrt(G*(M1+M2)*ai)
    Lo_e0 = mu_o*np.sqrt(G*(M1+M2+M3)*ao)
    
    ei = np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)
    eo = np.sqrt(eo_x**2. + eo_y**2. + eo_z**2.)
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    S1 = np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.)
    S2 = np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.)
    
    # directional quantities
    uLi_x = Li_x / (Li_e0 * eff_i)
    uLi_y = Li_y / (Li_e0 * eff_i)
    uLi_z = Li_z / (Li_e0 * eff_i)
    uLo_x = Lo_x / (Lo_e0 * eff_o)
    uLo_y = Lo_y / (Lo_e0 * eff_o)
    uLo_z = Lo_z / (Lo_e0 * eff_o)
    
    ji_x, ji_y, ji_z = Li_x/Li_e0, Li_y/Li_e0, Li_z/Li_e0
    jo_x, jo_y, jo_z = Lo_x/Lo_e0, Lo_y/Lo_e0, Lo_z/Lo_e0
    
    uS1_x, uS1_y, uS1_z = S1_x/S1, S1_y/S1, S1_z/S1
    uS2_x, uS2_y, uS2_z = S2_x/S2, S2_y/S2, S2_z/S2
    
    # get GR & GW terms
    y_orb_vect = np.array([\
        Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, ai\
                ])
    par_GR = np.array([mu_i, omega_i,\
                       Li_e0, ei, eff_i,\
                       uLi_x, uLi_y, uLi_z])
    
    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z, \
    dai \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        
    # get LK terms
    y_LK_vect = np.array([\
        Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
        Lo_x, Lo_y, Lo_z, eo_x, eo_y, eo_z, \
        ai])
    
    par_LK = np.array([mu_i, mu_o, omega_i, \
                       Li_e0, Lo_e0, ei, eo, eff_i, eff_o, \
                       ji_x, ji_y, ji_z,\
                       jo_x, jo_y, jo_z, uLo_x, uLo_y, uLo_z,\
                       dai])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, deo_LK_x, deo_LK_y, deo_LK_z, \
    __\
        = get_dy_LK_quad_da(y_LK_vect, par, par_LK)
        
    # get spin terms
    y_SP_vect = np.array([\
        Li_x, Li_y, Li_z, ei_x, ei_y, ei_z,\
        ai,\
        S1_x, S1_y, S1_z,\
        S2_x, S2_y, S2_z])
    
    par_SP = np.array([mu_i, omega_i, Li_e0, ei, eff_i, dai, \
                       uLi_x, uLi_y, uLi_z, \
                       S1, S2, \
                       uS1_x, uS1_y, uS1_z, uS2_x, uS2_y, uS2_z])
    
    dLi_SP_x, dLi_SP_y, dLi_SP_z, dei_SP_x, dei_SP_y, dei_SP_z, \
    __, \
    dS1_SP_x, dS1_SP_y, dS1_SP_z, \
    dS2_SP_x, dS2_SP_y, dS2_SP_z\
        = get_dy_SP(y_SP_vect, par, par_SP)
        
    # total 
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
    
    # GW of semi-major axis
    dai_nat = dai / ai_unit
    
    # S1/S2 sees SP (de Sitter & Lense-Thirring)
    dS1_nat_x = dS1_SP_x / S1_unit
    dS1_nat_y = dS1_SP_y / S1_unit
    dS1_nat_z = dS1_SP_z / S1_unit
    dS2_nat_x = dS2_SP_x / S2_unit
    dS2_nat_y = dS2_SP_y / S2_unit
    dS2_nat_z = dS2_SP_z / S2_unit
    
    dy_nat_vect = np.array([\
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, \
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, \
            dai_nat, \
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
    
    return dy_nat_vect

@jit(nopython=True, fastmath=True)
def evol_binary(t_nat, y_nat_vect, par):
    # parse parameters
    # 0-5
    # 6
    # 7-9
    # 10-12
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    ai_nat, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
        
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    M1, M2, __, __,\
    br_flag, ss_flag\
                = par
        
    # convert to cgs units
    Li_x, Li_y, Li_z = Li_nat_x*Li_unit, Li_nat_y*Li_unit, Li_nat_z*Li_unit
    ai = ai_nat * ai_unit
    
    S1_x, S1_y, S1_z = S1_nat_x*S1_unit, S1_nat_y*S1_unit, S1_nat_z*S1_unit
    S2_x, S2_y, S2_z = S2_nat_x*S2_unit, S2_nat_y*S2_unit, S2_nat_z*S2_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    Li_e0 = mu_i*np.sqrt(G*(M1+M2)*ai)

    ei = np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)
    eff_i = np.sqrt(1.-ei**2.)
    
    S1 = np.sqrt(S1_x**2. + S1_y**2. + S1_z**2.)
    S2 = np.sqrt(S2_x**2. + S2_y**2. + S2_z**2.)
    
    # directional quantities
    uLi_x = Li_x / (Li_e0 * eff_i)
    uLi_y = Li_y / (Li_e0 * eff_i)
    uLi_z = Li_z / (Li_e0 * eff_i)
    
    ji_x, ji_y, ji_z = Li_x/Li_e0, Li_y/Li_e0, Li_z/Li_e0
    
    uS1_x, uS1_y, uS1_z = S1_x/S1, S1_y/S1, S1_z/S1
    uS2_x, uS2_y, uS2_z = S2_x/S2, S2_y/S2, S2_z/S2
    
    # get GR & GW terms
    y_orb_vect = np.array([\
        Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, ai\
                ])
    par_GR = np.array([mu_i, omega_i,\
                       Li_e0, ei, eff_i,\
                       uLi_x, uLi_y, uLi_z])
    
    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z, \
    dai \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        
    
    # get spin terms
    y_SP_vect = np.array([\
        Li_x, Li_y, Li_z, ei_x, ei_y, ei_z,\
        ai,\
        S1_x, S1_y, S1_z,\
        S2_x, S2_y, S2_z])
    
    par_SP = np.array([mu_i, omega_i, Li_e0, ei, eff_i, dai, \
                       uLi_x, uLi_y, uLi_z, \
                       S1, S2, \
                       uS1_x, uS1_y, uS1_z, uS2_x, uS2_y, uS2_z])
    
    dLi_SP_x, dLi_SP_y, dLi_SP_z, dei_SP_x, dei_SP_y, dei_SP_z, \
    __, \
    dS1_SP_x, dS1_SP_y, dS1_SP_z, \
    dS2_SP_x, dS2_SP_y, dS2_SP_z\
        = get_dy_SP(y_SP_vect, par, par_SP)
        
    # total 
    # inner orb sees GR&GW + SP back reaction
    dLi_nat_x = (dLi_GR_x + dLi_SP_x) / Li_unit
    dLi_nat_y = (dLi_GR_y + dLi_SP_y) / Li_unit
    dLi_nat_z = (dLi_GR_z + dLi_SP_z) / Li_unit
    dei_x = dei_GR_x + dei_SP_x
    dei_y = dei_GR_y + dei_SP_y
    dei_z = dei_GR_z + dei_SP_z
    
    # GW of semi-major axis
    dai_nat = dai / ai_unit
    
    # S1/S2 sees SP (de Sitter & Lense-Thirring)
    dS1_nat_x = dS1_SP_x / S1_unit
    dS1_nat_y = dS1_SP_y / S1_unit
    dS1_nat_z = dS1_SP_z / S1_unit
    dS2_nat_x = dS2_SP_x / S2_unit
    dS2_nat_y = dS2_SP_y / S2_unit
    dS2_nat_z = dS2_SP_z / S2_unit
    
    dy_nat_vect = np.array([\
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, \
            dai_nat, \
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
#    print(dy_nat_vect)
    return dy_nat_vect
    
    
    
            
            
            
            

            