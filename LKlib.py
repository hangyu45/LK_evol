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
    
    G3muMt2_c5a3 = (G*mu/c**2./a_orb) * (G*Mt/c**2./a_orb)**2. * c
    
    inv_t_gw = (64./5.)*(G3muMt2_c5a3/a_orb) \
                * (1.+73./24.*e**2+37./96.*e**4.)/(1.-e**2.)**(3.5)
    t_gw = 1./inv_t_gw
    return t_gw

@jit(nopython=True, fastmath=True)
def get_t_lk(M1, M2, M3, ai, ao):
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    t_lk = 1./omega_i * (M1+M2)/M3 * (ao/ai)**3.
    return t_lk

def get_epsilon_GR(M1, M2, M3, ai, ao, eo=0):
    M12 = M1+M2
    ep_GR = 3.*(G*M12/c**2./ai) * (M12/M3)*(ao/ai)**3.
    return ep_GR

def get_epsilon_BR(M1, M2, M3, ai, ao, eo=0):
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    ep_BR=(mu_i/mu_o) * np.sqrt((M1+M2)/(M1+M2+M3) * (ai/ao/(1.-eo**2.)))
    return ep_BR

def find_ei_max_vs_Ii_0(Ii_0, \
        M1, M2, M3, ai, ao, eo=0):
    cI0 = np.cos(Ii_0)
    ep_GR = get_epsilon_GR(M1, M2, M3, ai, ao, eo)
    ep_BR = get_epsilon_BR(M1, M2, M3, ai, ao, eo)
    
    def resi(ei_max):
        jj = np.sqrt(1.-ei_max**2.)
        resi = 0.375 * ((jj+1)/jj)\
            * (5*(cI0+0.5*ep_BR)**2.\
               - (3.+4.*ep_BR*cI0+2.25*ep_BR**2.)*jj**2.\
               + (ep_BR**2.*jj**4.))\
            + ep_GR
        return resi

    ei_max = opt.ridder(resi, 0., 1.-1.e-18)
    return ei_max

def find_ei_lim_over_Ii_0(M1, M2, M3, ai, ao, eo=0):
    ep_GR = get_epsilon_GR(M1, M2, M3, ai, ao, eo)
    ep_BR = get_epsilon_BR(M1, M2, M3, ai, ao, eo)
    
    def resi(ei_lim):
        jj = np.sqrt(1.-ei_lim**2.)
        resi = 0.375*(jj+1.)*jj\
                * (-3. + 0.25*ep_BR**2.*(0.8*jj**2.-1))\
            + ep_GR
        return resi
    
    ei_lim = opt.ridder(resi, 0., 1.-1.e-18)
    return ei_lim

def find_Ii_lim_for_ei_lim(M1, M2, M3, ai, ao, eo=0):
    ei_lim = find_ei_lim_over_Ii_0(M1, M2, M3, ai, ao, eo)
    ep_BR = get_epsilon_BR(M1, M2, M3, ai, ao, eo)
    
    j_lim_sq = 1.-ei_lim**2.
    cIi_lim = 0.5*ep_BR*(0.8*j_lim_sq - 1.)
    Ii_lim = np.arccos(cIi_lim)
    return Ii_lim

def find_Tgw_min_vs_Ii_0(Ii_0, \
        M1, M2, M3, ai, ao, eo=0):
    ei_max = find_ei_max_vs_Ii_0(Ii_0, \
        M1, M2, M3, ai, ao, eo)
    
    # note t_gw_min is not simply given by instant. merger time at e_max
    # but needs 1/\sqrt(1-e**2) more time
    t_gw_min = get_inst_t_gw_from_a_orb(M1, M2, ai, 0)
    t_gw_min *= (1.-ei_max**2.)**3.
    return t_gw_min

def find_Tgw_lim_over_Ii_0(M1, M2, M3, ai, ao, eo=0):
    ei_lim = find_ei_lim_over_Ii_0(M1, M2, M3, ai, ao, eo)
    t_gw_lim = get_inst_t_gw_from_a_orb(M1, M2, ai, 0)
    t_gw_lim *= (1.-ei_lim**2.)**3.
    return t_gw_lim

def find_merger_window(Tgw_trgt, \
                      M1, M2, M3, ai, ao, eo=0.):
    def resi(Ii_0):
        Tgw = find_Tgw_min_vs_Ii_0(Ii_0, \
            M1, M2, M3, ai, ao, eo)
        resi = 1.-Tgw/Tgw_trgt
        return resi
    ep_GR = get_epsilon_GR(M1, M2, M3, ai, ao, eo)
    ep_BR = get_epsilon_BR(M1, M2, M3, ai, ao, eo)
    
    e_lim = find_ei_lim_over_Ii_0(M1, M2, M3, ai, ao, eo)
    j_lim = np.sqrt(1.-e_lim**2.)
    I_lim = 0.5*ep_BR*(0.8*j_lim**2.-1.)
    I_lim = np.arccos(I_lim)
    
    I_mm = 0.1*(-ep_BR\
                +np.sqrt(ep_BR**2.+60.-80.*ep_GR/3.))
    I_mm = np.arccos(I_mm)
    I_pp = 0.1*(-ep_BR\
                -np.sqrt(ep_BR**2.+60.-80.*ep_GR/3.))
    I_pp = np.arccos(I_pp)
#     print(I_mm*180./np.pi, I_pp*180./np.pi)
    I_m = opt.ridder(resi, I_mm*(1.+1.e-12), I_lim)
    I_p = opt.ridder(resi, I_lim, I_pp*(1.-1.e-12))
    return I_m, I_p

def check_LK_exc(M1, M2, M3, ai, ao, eo=0):
    ep_GR = get_epsilon_GR(M1, M2, M3, ai, ao, eo)
    ep_BR = get_epsilon_BR(M1, M2, M3, ai, ao, eo)
    
    flag = 2.25+3.*ep_BR**2./80. - ep_GR
    return flag

def check_stability(M1, M2, M3, ai, ao, eo=0):
    flag =ao/ai - 2.8*(1.-0.3*0.5)*(1.+M3/(M1+M2))**0.4*(1.+eo)**0.4/(1.-eo)**1.2
    return flag

def check_DA_given_ei_max(ei_max, \
                          M1, M2, M3, ai, ao, eo=0):
    M12 = M1+M2
    M123 = M1+M2+M3
    
    omega_i = np.sqrt(G*M12/ai**3.)
    omega_o = np.sqrt(G*M123/ao**3.)
    
    Tlk = 1./omega_i * (M12/M3) * (ao*np.sqrt(1.-eo**2.)/ai)**3.
    flag = Tlk * np.sqrt(1.-ei_max**2.) - 2.*np.pi/omega_o
    return flag


@jit(nopython=True, fastmath=True)
def get_dy_orb_GR_GW(y_orb_vect, par, par_GR, fudge_gw=1.):
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
    
    G3muMt2_c5a3 = (G*mu_i/c**2./ai) * (G*(M1+M2)/c**2./ai)**2. * c
    ei2 = ei**2.
    ei4 = ei**4.
    dai = - (64./5.) * (G3muMt2_c5a3)\
          * (1. + 73./24.*ei2 + 37./96.*ei4)/(eff_i**7.)    
    dLi_GW = - (32./5.)*G3muMt2_c5a3*np.sqrt(G) * mu_i*np.sqrt(M1+M2) / np.sqrt(ai) \
          * (1.+0.875*ei2)/(eff_i**4.)
    dei_GW = - (304./15.) * G3muMt2_c5a3/ai\
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
    
    dai *= fudge_gw
    dLi_GW_v *= fudge_gw
    dei_GW_v *= fudge_gw
        
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
def get_dy_orb_noAvg(y_orb_vect, M1, M2):
    """
    cf. eq. (2.3) of Lincoln & Will
    Note that v_vect is not purely tangitial (corresponding to the r*dphidt component) 
    but contains some radial components due to drdt term
    """
    rx, ry, rz, \
    vx, vy, vz\
        = y_orb_vect
    
    eta = M1*M2/(M1+M2)**2.
    r_v = np.array([rx, ry, rz])
    v_v = np.array([vx, vy, vz])
    
    r_sq = rx**2. + ry**2. + rz**2.
    v_sq = vx**2. + vy**2. + vz**2.
    rr = np.sqrt(r_sq)
    vv = np.sqrt(v_sq)
    
    drdt = (rx*vx+ry*vy+rz*vz)/rr
    rdphidt = np.sqrt(v_sq-drdt**2.)
    
    drdt /= c
    rdphidt /= c
    v_sq /= c**2.
    
    drdt_sq=drdt**2.
    M_r= G*(M1+M2)/rr/c**2.
    acc_unit = G*(M1+M2)/r_sq
    
    A1 = 2.*(2.+eta)*M_r\
        - (1.+3*eta)*v_sq\
        + 1.5*eta*drdt_sq
    A2 = - 0.75*(12.+29.*eta)*M_r**2.\
         - eta * (3.-4.*eta)*v_sq**2.\
         - 15./8.*eta*(1.-3.*eta)*drdt_sq**2.\
         + 1.5*eta*(3.-4.*eta)*v_sq*drdt_sq\
         + 0.5*eta*(13.-4.*eta)*M_r*v_sq \
         + (2.+25.*eta+2.*eta**2.)*M_r*drdt_sq
        
    A2p5 = 8./5.*eta*M_r*drdt\
        * (3.*v_sq + 17./3.*M_r)
    AA = A1 + A2 + A2p5
        
    B1 = 2.*(2.-eta)*drdt
    B2 = 0.5*drdt*(\
            eta*(15.+4.*eta)*v_sq\
          - (4.+41.*eta+8.*eta**2.)*M_r\
          - 3.*eta*(3.+2.*eta)*drdt_sq )
    B2p5 = -8./5.*eta*M_r*(6*v_sq-2*M_r-15.*drdt_sq)
    BB = B1 + B2 + B2p5
    
    acc_ur = acc_unit * (-1. + AA)
    acc_uv= acc_unit * BB
    
    a_v = acc_ur*r_v/rr + acc_uv*v_v/c
       
    dy_orb_vect = np.array([\
                    vx, vy, vz, 
                    a_v[0], a_v[1], a_v[2]
                           ])
    return dy_orb_vect

@jit(nopython=True)
def get_acc_GR(Kep_vect, par):
    M1, M2, t_unit, r_unit=par
    eta = M1 * M2/(M1 + M2)**2.
    
    r_nat, drdt_nat, phi, dphidt_nat=Kep_vect
    r = r_nat * r_unit
    drdt = drdt_nat * r_unit / t_unit
    dphidt = dphidt_nat / t_unit
    rdphidt = r*dphidt
    v_sq = drdt**2. + rdphidt**2.
    
    drdt /= c
    rdphidt /= c
    v_sq /= c**2.
    
    drdt_sq=drdt**2.
    M_r= G*(M1+M2)/r/c**2.
    acc_unit = G*(M1+M2)/r**2.
    
    A1 = 2.*(2.+eta)*M_r\
        - (1.+3*eta)*v_sq\
        + 1.5*eta*drdt_sq
    A2 = - 0.75*(12.+29.*eta)*M_r**2.\
         - eta * (3.-4.*eta)*v_sq**2.\
         - 15./8.*eta*(1.-3.*eta)*drdt_sq**2.\
         + 1.5*eta*(3.-4.*eta)*v_sq*drdt_sq\
         + 0.5*eta*(13.-4.*eta)*M_r*v_sq \
         + (2.+25.*eta+2.*eta**2.)*M_r*drdt_sq
        
    A2p5 = 8./5.*eta*M_r*drdt\
        * (3.*v_sq + 17./3.*M_r)
    AA = A1 + A2 + A2p5
        
    B1 = 2.*(2.-eta)*drdt
    B2 = 0.5*drdt*(\
            eta*(15.+4.*eta)*v_sq\
          - (4.+41.*eta+8.*eta**2.)*M_r\
          - 3.*eta*(3.+2.*eta)*drdt_sq )
#     B2p5 = -8./5.*eta*M_r*(v_sq + 3.*M_r)
    B2p5 = -8./5.*eta*M_r*(6*v_sq-2*M_r-15.*drdt_sq)
    BB = B1 + B2 + B2p5
    
    acc_r = acc_unit * (AA + drdt*BB)
    acc_phi = acc_unit * (rdphidt*BB)
    return acc_r, acc_phi

@jit(nopython=True)
def orb_evol(t_nat, Kep_vect, \
             acc_r, acc_phi, par):
    M1, M2, t_unit, r_unit=par
    GMt=G*(M1+M2)
    
    r_nat, drdt_nat, phi, dphidt_nat=Kep_vect
    r = r_nat * r_unit
    drdt = drdt_nat * r_unit / t_unit
    dphidt = dphidt_nat / t_unit
    
    ddr = r * dphidt**2. - GMt/r**2. + acc_r
    ddr_nat = ddr * t_unit**2./r_unit
    
    ddphi = -2.*drdt*dphidt/r + acc_phi/r
    ddphi_nat = ddphi * t_unit**2.
    
    dKep=np.array([drdt_nat, ddr_nat, dphidt_nat, ddphi_nat])
    return dKep

@jit(nopython=True)
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
    
#     ji_d_uLo = inner(ji_v, uLo_v)
#     ei_d_uLo = inner(ei_v, uLo_v)
    ji_d_uLo=(uLi_x*uLo_x+uLi_y*uLo_y+uLi_z*uLo_z)*eff_i
    ei_d_uLo=(ei_x*uLo_x+ei_y*uLo_y+ei_z*uLo_z)
    
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

@jit(nopython=True)
def get_dy_LK_quad_sa(y_LK_vect, par, par_LK):
    """
    Lidov-Kozai under single averaging
    """
    # parse input; note that the kep elements for the outer binary is different
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
    ro_x, ro_y, ro_z, vo_x, vo_y, vo_z\
                = y_LK_vect
    
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
    
    # par for LK calc
    mu_i, mu_o, omega_i, ai, \
    Li_e0, ei, eff_i, \
    uLi_x, uLi_y, uLi_z \
                = par_LK
    
    # scalar quantities
    ro = np.sqrt(ro_x**2. + ro_y**2. + ro_z**2.)
    t_LK = (1./omega_i) * (M1+M2)/M3 * (ro/ai)**3.
    inv_t_LK_1p5 = 1.5/t_LK
    
    phiK = G * (M1+M2)*M3/mu_o / ro
    phiQ = 0.25*G*M3/ro * (mu_i/mu_o) * (ai/ro)**2.
    
    # directional products
    ji_v  = np.array([uLi_x, uLi_y, uLi_z]) * eff_i
    ei_v  = np.array([ei_x,  ei_y,  ei_z]) 
    
    uro_v = np.array([ro_x, ro_y, ro_z])/ro
    
#     ji_d_uro = inner(ji_v, uro_v)
#     ei_d_uro = inner(ei_v, uro_v)
    ji_d_uro = (uLi_x*ro_x+uLi_y*ro_y+uLi_z*ro_z)*eff_i/ro
    ei_d_uro = (ei_x*ro_x+ei_y*ro_y+ei_z*ro_z)/ro
    
    ji_c_uro_v = cross(ji_v, uro_v)
    ei_c_uro_v = cross(ei_v, uro_v)
    ji_c_ei_v = cross(ji_v, ei_v)
    
    ro_grad_inv_ro = - uro_v/ro
    grad_ji_d_uro = ji_d_uro*ro_grad_inv_ro + ji_v/ro
    grad_ei_d_uro = ei_d_uro*ro_grad_inv_ro + ei_v/ro
    
    # derivatives of inner orbit's dir vects 
    dji_v = inv_t_LK_1p5 * (5.*ei_d_uro*ei_c_uro_v - ji_d_uro*ji_c_uro_v)
    dei_v = inv_t_LK_1p5 * (5.*ei_d_uro*ji_c_uro_v - ji_d_uro*ei_c_uro_v - 2.*ji_c_ei_v)
    
    dLi_v = Li_e0 * dji_v
    
    # Keplerian modtions of the outer binary
    acco_v = phiK * ro_grad_inv_ro \
            -phiQ* (3.*ro_grad_inv_ro*(-1. + 6.*ei**2.+3.*ji_d_uro**2. - 15.*ei_d_uro**2.)\
                  + 6. * ji_d_uro*grad_ji_d_uro \
                  - 30.* ei_d_uro*grad_ei_d_uro)
    
    dy_LK_vect = np.array([\
                 dLi_v[0],  dLi_v[1],  dLi_v[2],\
                 dei_v[0],  dei_v[1],  dei_v[2],\
                 vo_x,      vo_y,      vo_z,    \
                 acco_v[0], acco_v[1], acco_v[2]\
        ])
    
    return dy_LK_vect
    

@jit(nopython=True, fastmath=True)
def get_dy_SP(y_SP_vect, par, par_SP):
    """
    de Sitter spin-orbit & Lense Thirring spin-spin & quadrupole-monopole coupling
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
    
    omega1_QM = 0.5*G_c2*S1*(M2/M1)/(ai3_eff_i3)
    omega2_QM = 0.5*G_c2*S2*(M1/M2)/(ai3_eff_i3)
    omega1_QM_br = -0.75*G_c2*S1**2.*(M2/M1)/(ai3_eff_i3*Li_e0*eff_i)
    omega2_QM_br = -0.75*G_c2*S2**2.*(M1/M2)/(ai3_eff_i3*Li_e0*eff_i)
    
    omega1_QM_S1 = omega1_QM * S1 * ss_flag
    omega2_QM_S2 = omega2_QM * S2 * ss_flag
    omega1_QM_br_Li = omega1_QM_br * Li_e0*eff_i * ss_flag * br_flag
    omega2_QM_br_Li = omega2_QM_br * Li_e0*eff_i * ss_flag * br_flag
    
    # directional products 
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x,  ei_y,  ei_z ])
    uS1_v = np.array([uS1_x, uS1_y, uS1_z])
    uS2_v = np.array([uS2_x, uS2_y, uS2_z])
    
#     uLi_d_uS1 = inner(uLi_v, uS1_v)
#     uLi_d_uS2 = inner(uLi_v, uS2_v)
#     uS1_d_uS2 = inner(uS1_v, uS2_v)
    uLi_d_uS1 = (uLi_x*uS1_x + uLi_y*uS1_y + uLi_z*uS1_z)
    uLi_d_uS2 = (uLi_x*uS2_x + uLi_y*uS2_y + uLi_z*uS2_z)
    uS1_d_uS2 = (uS1_x*uS2_x + uS1_y*uS2_y + uS1_z*uS2_z)
    
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
    dLi_SS_v = omega_SS_br_Li * (- uLi_d_uS1*uLi_c_uS2_v - uLi_d_uS2*uLi_c_uS1_v)
    dei_SS_v = omega_SS_br * (uLi_d_uS1*uS2_c_ei_v + uLi_d_uS2*uS1_c_ei_v\
                             +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_v)
    dS1_SS_v = omega1_SS_S1 * (- uS1_c_uS2_v - 3.*uLi_d_uS2*uLi_c_uS1_v)
    dS2_SS_v = omega2_SS_S2 * (+ uS1_c_uS2_v - 3.*uLi_d_uS1*uLi_c_uS2_v)
    
    # quad-mono
    dLi_QM_v = omega1_QM_br_Li * (-2.*uLi_d_uS1*uLi_c_uS1_v) \
             + omega2_QM_br_Li * (-2.*uLi_d_uS2*uLi_c_uS2_v)
    dei_QM_v = omega1_QM_br * (-2.*uLi_d_uS1*uLi_c_uS1_v + (1.-5.*uLi_d_uS1**2.)*uLi_c_ei_v)\
             + omega2_QM_br * (-2.*uLi_d_uS2*uLi_c_uS2_v + (1.-5.*uLi_d_uS2**2.)*uLi_c_ei_v)
    
    dS1_QM_v = omega1_QM_S1 * (-3.*uLi_d_uS1*uLi_c_uS1_v)
    dS2_QM_v = omega2_QM_S2 * (-3.*uLi_d_uS2*uLi_c_uS2_v)
    
    # total 
    dLi_v = dLi_SL_v + dLi_SS_v + dLi_QM_v
    dei_v = dei_SL_v + dei_SS_v + dei_QM_v
    dS1_v = dS1_SL_v + dS1_SS_v + dS1_QM_v
    dS2_v = dS2_SL_v + dS2_SS_v + dS2_QM_v
    
    dy_SP_vect = np.array([\
                 dLi_v[0], dLi_v[1], dLi_v[2], \
                 dei_v[0], dei_v[1], dei_v[2], \
                 dS1_v[0], dS1_v[1], dS1_v[2], \
                 dS2_v[0], dS2_v[1], dS2_v[2]])
    return dy_SP_vect    
    
    
# @jit(nopython=True, fastmath=True)
# def get_dy_SP_backup(y_SP_vect, par, par_SP):
#     """
#     de Sitter spin-orbit & Lense Thirring spin-spin
#     """
#     # parse input
#     Li_x, Li_y, Li_z, \
#     ei_x, ei_y, ei_z, \
#     S1_x, S1_y, S1_z, \
#     S2_x, S2_y, S2_z \
#         = y_SP_vect
        
#     # global par
#     M1, M2, M3, ao,\
#     t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
#     br_flag, ss_flag\
#                 = par
        
#     # par for LK calc
#     mu_i, omega_i, ai, \
#     Li_e0, ei, eff_i, S1, S2, \
#     uLi_x, uLi_y, uLi_z, \
#     uS1_x, uS1_y, uS1_z, uS2_x, uS2_y, uS2_z\
#                 = par_SP
        
#     # scalar quantities
#     G_c2 = G/c**2.
#     ai3_eff_i3 = (ai*eff_i)**3.
#     omega1_SL = 1.5*G_c2*(M2+mu_i/3.)/(ai*eff_i**2.)*omega_i
#     omega2_SL = 1.5*G_c2*(M1+mu_i/3.)/(ai*eff_i**2.)*omega_i
#     omega1_SL_br = 0.5*G_c2*S1*(4.+3.*M2/M1)/(ai3_eff_i3)
#     omega2_SL_br = 0.5*G_c2*S2*(4.+3.*M1/M2)/(ai3_eff_i3)
    
#     omega1_SL_S1 = omega1_SL * S1
#     omega2_SL_S2 = omega2_SL * S2
#     omega1_SL_br_Li = omega1_SL_br*Li_e0*eff_i * br_flag
#     omega2_SL_br_Li = omega2_SL_br*Li_e0*eff_i * br_flag
    
#     omega1_SS = 0.5*G_c2*S2/(ai3_eff_i3)
#     omega2_SS = 0.5*G_c2*S1/(ai3_eff_i3)
#     omega_SS_br = -1.5*G_c2*S1*S2/(mu_i*omega_i)/(ai3_eff_i3*ai**2.*eff_i)
    
#     omega1_SS_S1 = omega1_SS * S1 * ss_flag
#     omega2_SS_S2 = omega2_SS * S2 * ss_flag
#     omega_SS_br_Li = omega_SS_br*Li_e0*eff_i * ss_flag * br_flag
    
#     # directional products 
#     uLi_v = np.array([uLi_x, uLi_y, uLi_z])
#     ei_v  = np.array([ei_x,  ei_y,  ei_z ])
#     uS1_v = np.array([uS1_x, uS1_y, uS1_z])
#     uS2_v = np.array([uS2_x, uS2_y, uS2_z])
    
# #     uLi_d_uS1 = inner(uLi_v, uS1_v)
# #     uLi_d_uS2 = inner(uLi_v, uS2_v)
# #     uS1_d_uS2 = inner(uS1_v, uS2_v)
#     uLi_d_uS1 = (uLi_x*uS1_x + uLi_y*uS1_y + uLi_z*uS1_z)
#     uLi_d_uS2 = (uLi_x*uS2_x + uLi_y*uS2_y + uLi_z*uS2_z)
#     uS1_d_uS2 = (uS1_x*uS2_x + uS1_y*uS2_y + uS1_z*uS2_z)
    
#     uLi_c_uS1_v = cross(uLi_v, uS1_v)
#     uLi_c_uS2_v = cross(uLi_v, uS2_v)
#     uS1_c_uS2_v = cross(uS1_v, uS2_v)
#     uS1_c_ei_v  = cross(uS1_v, ei_v )
#     uS2_c_ei_v  = cross(uS2_v, ei_v )
#     uLi_c_ei_v  = cross(uLi_v, ei_v ) 

#     # de Sitter
#     dLi_SL_v = omega1_SL_br_Li * (-uLi_c_uS1_v)\
#              + omega2_SL_br_Li * (-uLi_c_uS2_v)
#     dei_SL_v = omega1_SL_br * (uS1_c_ei_v - 3.*uLi_d_uS1*uLi_c_ei_v)\
#              + omega2_SL_br * (uS2_c_ei_v - 3.*uLi_d_uS2*uLi_c_ei_v)
#     dS1_SL_v = omega1_SL_S1 * (uLi_c_uS1_v)
#     dS2_SL_v = omega2_SL_S2 * (uLi_c_uS2_v)
    
#     # Lense-Thirring
#     dLi_SS_v = omega_SS_br_Li * (- uLi_d_uS1*uLi_c_uS2_v - uLi_d_uS2*uLi_c_uS1_v)
#     dei_SS_v = omega_SS_br * (uLi_d_uS1*uS2_c_ei_v + uLi_d_uS2*uS1_c_ei_v\
#                              +(uS1_d_uS2 - 5.*uLi_d_uS1*uLi_d_uS2)*uLi_c_ei_v)
#     dS1_SS_v = omega1_SS_S1 * (- uS1_c_uS2_v - 3.*uLi_d_uS2*uLi_c_uS1_v)
#     dS2_SS_v = omega2_SS_S2 * (+ uS1_c_uS2_v - 3.*uLi_d_uS1*uLi_c_uS2_v)
    
#     # total 
#     dLi_v = dLi_SL_v + dLi_SS_v
#     dei_v = dei_SL_v + dei_SS_v
#     dS1_v = dS1_SL_v + dS1_SS_v
#     dS2_v = dS2_SL_v + dS2_SS_v
    
#     dy_SP_vect = np.array([\
#                  dLi_v[0], dLi_v[1], dLi_v[2], \
#                  dei_v[0], dei_v[1], dei_v[2], \
#                  dS1_v[0], dS1_v[1], dS1_v[2], \
#                  dS2_v[0], dS2_v[1], dS2_v[2]])
#     return dy_SP_vect

@jit(nopython=True, fastmath=True)
def get_dy_SMBH_da(y_SMBH_vect, par, par_SMBH):
    """
    GR effects associated w/ SMBH
    """
    # parse input
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
    Lo_x, Lo_y, Lo_z, eo_x, eo_y, eo_z, \
    S1_x, S1_y, S1_z, S2_x, S2_y, S2_z \
                = y_SMBH_vect
        
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, Lo_unit, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
        
    # use same set of par for the LK evol
    mu_i, mu_o, omega_i, ai, \
    Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
    S1, S2,\
    uLi_x, uLi_y, uLi_z, \
    uLo_x, uLo_y, uLo_z, \
    uS1_x, uS1_y, uS1_z, \
    uS2_x, uS2_y, uS2_z, \
    smbh_flag, I_S3, phi_S3\
                = par_SMBH

    # scalar quantities
    ### FIXME ###
    S3 = G*M3**2./c # assuming maximally spinning SMBH
    
    G_c2_ao = G/(c**2.*ao)
    omega_o = np.sqrt(G*(M1+M2+M3)/ao**3.)
    Li = Li_e0 * eff_i
    Lo = Lo_e0 * eff_i
    
    omega_LT_S3 = 0.5 * G_c2_ao*S3*(4.+3.*(M1+M2)/M3)/(ao**2.*(1.-eo**2.)**1.5)
    omega_dS_io = 1.5 * G_c2_ao*(M3+mu_o/3.)*omega_o/(1.-eo**2.)
    omega_dS_SLo = omega_dS_io
    
    omega_LT_dir = 0.5 * G_c2_ao*S3/(ao**2.*(1.-eo**2.)**1.5)
    
    # directional products 
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x,  ei_y,  ei_z ])
    uLo_v = np.array([uLo_x, uLo_y, uLo_z])
    eo_v  = np.array([eo_x,  eo_y,  eo_z])
    uS1_v = np.array([uS1_x, uS1_y, uS1_z])
    uS2_v = np.array([uS2_x, uS2_y, uS2_z])
    
    ### FIXME ###
    uS3_v = np.array([np.sin(I_S3)*np.cos(phi_S3), 
                      np.sin(I_S3)*np.sin(phi_S3),
                      np.cos(I_S3)]) 
    # Lo/S3 ~ 5e-6; ignoring backreaction onto S3
    
    uLo_d_uS3 = inner(uLo_v, uS3_v)
    
    uS3_c_uLo = cross(uS3_v, uLo_v)
    uS3_c_uLi = cross(uS3_v, uLi_v)
    uS3_c_uS1 = cross(uS3_v, uS1_v)
    uS3_c_uS2 = cross(uS3_v, uS2_v)
    uLo_c_uLi = cross(uLo_v, uLi_v)
    uLo_c_ei  = cross(uLo_v, ei_v)
    uLo_c_uS1 = cross(uLo_v, uS1_v)
    uLo_c_uS2 = cross(uLo_v, uS2_v)
    
    # effect 1: LT (in fact, dS br) of Lo around S3; ignoring back-reaction on S3
    dLo_v = Lo * omega_LT_S3 * uS3_c_uLo
    ### fixme ###
    deo_v = np.zeros(3) # fix circular outer orbit
    
    # effect 2: dS of Li around Lo; ignoring back-reaction on Lo
    dLi_v_dS = Li * omega_dS_io * uLo_c_uLi
    dei_v    = omega_dS_io * uLo_c_ei
    
    # effect 3: dS of S1, S2 around Lo; ignoring back-reaction on Lo
    dS1_v_dS = S1 * omega_dS_SLo * uLo_c_uS1 
    dS2_v_dS = S2 * omega_dS_SLo * uLo_c_uS2 
    
    # effect 4: LT of Li, S1, S2 around S3
    dLi_v_LT = Li * omega_LT_dir * (uS3_c_uLi - 3.*uLo_d_uS3*uLo_c_uLi)
    dS1_v_LT = S1 * omega_LT_dir * (uS3_c_uS1 - 3.*uLo_d_uS3*uLo_c_uS1)
    dS2_v_LT = S2 * omega_LT_dir * (uS3_c_uS2 - 3.*uLo_d_uS3*uLo_c_uS2)
    
    dLi_v = dLi_v_dS + dLi_v_LT
    dS1_v = dS1_v_dS + dS1_v_LT
    dS2_v = dS2_v_dS + dS2_v_LT
    
    # total 
    dy_SMBH_vect = np.array([\
        dLi_v[0], dLi_v[1], dLi_v[2], dei_v[0], dei_v[1], dei_v[2], \
        dLo_v[0], dLo_v[1], dLo_v[2], deo_v[0], deo_v[1], deo_v[2], \
        dS1_v[0], dS1_v[1], dS1_v[2], dS2_v[0], dS2_v[1], dS2_v[2]\
    ])
    
    dy_SMBH_vect *= smbh_flag
    
    return dy_SMBH_vect

@jit(nopython=True, fastmath=True)
def evol_LK_quad_da(t_nat, y_nat_vect, par, \
                    smbh_flag=0, I_S3 = np.pi/6, phi_S3 = 0.):
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
    Li = np.sqrt(Li_v[0]**2.+Li_v[1]**2.+Li_v[2]**2.)
    
    ei = np.sqrt(ei_v[0]**2.+ei_v[1]**2.+ei_v[2]**2.)
    eo = np.sqrt(eo_v[0]**2.+eo_v[1]**2.+eo_v[2]**2.)
    eff_i = Li/Li_e0
    eff_o = np.sqrt(1.-eo**2.)
    
    S1 = np.sqrt(S1_v[0]**2.+S1_v[1]**2.+S1_v[2]**2.)
    S2 = np.sqrt(S2_v[0]**2.+S2_v[1]**2.+S2_v[2]**2.)
    
    # unity vectors
    uLi_v = Li_v / Li
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
    
    
    # get the SMBH related GR effects
    y_SMBH_vect = np.array([Li_v[0], Li_v[1], Li_v[2], \
                            ei_v[0], ei_v[1], ei_v[2], \
                            Lo_v[0], Lo_v[1], Lo_v[2], \
                            eo_v[0], eo_v[1], eo_v[2], \
                            S1_v[0], S1_v[1], S1_v[2], \
                            S2_v[0], S2_v[1], S2_v[2]])
    par_SMBH = np.array([mu_i, mu_o, omega_i, ai, \
                        Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
                        S1, S2, \
                        uLi_v[0], uLi_v[1], uLi_v[2], \
                        uLo_v[0], uLo_v[1], uLo_v[2], \
                        uS1_v[0], uS1_v[1], uS1_v[2], \
                        uS2_v[0], uS2_v[1], uS2_v[2], 
                        smbh_flag, I_S3, phi_S3])
    
    dLi_SMBH_x, dLi_SMBH_y, dLi_SMBH_z, \
    dei_SMBH_x, dei_SMBH_y, dei_SMBH_z, \
    dLo_SMBH_x, dLo_SMBH_y, dLo_SMBH_z, \
    deo_SMBH_x, deo_SMBH_y, deo_SMBH_z, \
    dS1_SMBH_x, dS1_SMBH_y, dS1_SMBH_z, \
    dS2_SMBH_x, dS2_SMBH_y, dS2_SMBH_z\
        = get_dy_SMBH_da(y_SMBH_vect, par, par_SMBH)
    
            
    # total 
    # GW of semi-major axis
    dai_nat = dai / ai_unit
    
    # inner orb sees GR&GW + LK + SP back reaction + SMBH effects
    dLi_nat_x = (dLi_GR_x + dLi_LK_x + dLi_SP_x + dLi_SMBH_x) / Li_unit
    dLi_nat_y = (dLi_GR_y + dLi_LK_y + dLi_SP_y + dLi_SMBH_y) / Li_unit
    dLi_nat_z = (dLi_GR_z + dLi_LK_z + dLi_SP_z + dLi_SMBH_z) / Li_unit
    dei_x = dei_GR_x + dei_LK_x + dei_SP_x + dei_SMBH_x
    dei_y = dei_GR_y + dei_LK_y + dei_SP_y + dei_SMBH_y
    dei_z = dei_GR_z + dei_LK_z + dei_SP_z + dei_SMBH_z
    
    # outer orb sees only LK & SMBH effects
    dLo_nat_x = (dLo_LK_x + dLo_SMBH_x) / Lo_unit 
    dLo_nat_y = (dLo_LK_y + dLo_SMBH_y) / Lo_unit
    dLo_nat_z = (dLo_LK_z + dLo_SMBH_z) / Lo_unit
    deo_x = deo_LK_x + deo_SMBH_x
    deo_y = deo_LK_y + deo_SMBH_y
    deo_z = deo_LK_z + deo_SMBH_z
    
    # S1/S2 sees SP (de Sitter & Lense-Thirring)
    dS1_nat_x = (dS1_SP_x + dS1_SMBH_x) / S1_unit
    dS1_nat_y = (dS1_SP_y + dS1_SMBH_y) / S1_unit
    dS1_nat_z = (dS1_SP_z + dS1_SMBH_z) / S1_unit
    dS2_nat_x = (dS2_SP_x + dS2_SMBH_x) / S2_unit
    dS2_nat_y = (dS2_SP_y + dS2_SMBH_y) / S2_unit
    dS2_nat_z = (dS2_SP_z + dS2_SMBH_z) / S2_unit
    
    dy_nat_vect = np.array([\
            dai_nat, \
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, \
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, \
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
    return dy_nat_vect

@jit(nopython=True, fastmath=True)
def evol_LK_quad_sa(t_nat, y_nat_vect, par):
    # parse parameters
    # 0
    # 1-6
    # 7-12
    # 13-15
    # 16-18
    ai_nat, \
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    ro_nat_x, ro_nat_y, ro_nat_z, vo_nat_x, vo_nat_y, vo_nat_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    
    # global par
    M1, M2, M3, ao,\
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
    
    # convert to cgs units
    ai = ai_nat * ai_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    S2_v = np.array([S2_nat_x, S2_nat_y, S2_nat_z]) * S2_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    ro_v = np.array([ro_nat_x, ro_nat_y, ro_nat_z]) * ao
    vo_v = np.array([vo_nat_x, vo_nat_y, vo_nat_z]) * ao/t_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    Li_e0 = mu_i*np.sqrt(G*(M1+M2)*ai)
    Li = np.sqrt(Li_v[0]**2.+Li_v[1]**2.+Li_v[2]**2.)
    ei = np.sqrt(ei_v[0]**2.+ei_v[1]**2.+ei_v[2]**2.)
    eff_i = Li/Li_e0
    
    S1 = np.sqrt(S1_v[0]**2.+S1_v[1]**2.+S1_v[2]**2.)
    S2 = np.sqrt(S2_v[0]**2.+S2_v[1]**2.+S2_v[2]**2.)
    
    # unity vectors
    uLi_v = Li_v / Li
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
                          ro_v[0], ro_v[1], ro_v[2], \
                          vo_v[0], vo_v[1], vo_v[2]])
    par_LK = np.array([mu_i, mu_o, omega_i, ai, \
                        Li_e0, ei, eff_i, \
                        uLi_v[0], uLi_v[1], uLi_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dro_LK_x, dro_LK_y, dro_LK_z, \
    dvo_LK_x, dvo_LK_y, dvo_LK_z\
        = get_dy_LK_quad_sa(y_LK_vect, par, par_LK)
        
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
    dro_nat_x = dro_LK_x / ao
    dro_nat_y = dro_LK_y / ao
    dro_nat_z = dro_LK_z / ao
    dvo_nat_x = dvo_LK_x / ao * t_unit
    dvo_nat_y = dvo_LK_y / ao * t_unit
    dvo_nat_z = dvo_LK_z / ao * t_unit
    
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
            dro_nat_x, dro_nat_y, dro_nat_z, dvo_nat_x, dvo_nat_y, dvo_nat_z, \
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
    return dy_nat_vect

@jit(nopython=True, fastmath=True)
def evol_binary(t_nat, y_nat_vect, par, fudge_gw=1.):
    # parse parameters
    # 0
    # 1-6
    # 7-9
    # 10-12
    ai_nat, \
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat_x, S2_nat_y, S2_nat_z\
                = y_nat_vect
    
    # global par
    M1, M2, M3, __,\
    t_unit, Li_unit, __, ai_unit, S1_unit, S2_unit, \
    br_flag, ss_flag\
                = par
    
    # convert to cgs units
    ai = ai_nat * ai_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    S2_v = np.array([S2_nat_x, S2_nat_y, S2_nat_z]) * S2_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    Li_e0 = mu_i*np.sqrt(G*(M1+M2)*ai)
    Li = np.sqrt(Li_v[0]**2.+Li_v[1]**2.+Li_v[2]**2.)
    ei = np.sqrt(ei_v[0]**2.+ei_v[1]**2.+ei_v[2]**2.)
    eff_i = Li/Li_e0
    
    S1 = np.sqrt(S1_v[0]**2.+S1_v[1]**2.+S1_v[2]**2.)
    S2 = np.sqrt(S2_v[0]**2.+S2_v[1]**2.+S2_v[2]**2.)
    
    # unity vectors
    uLi_v = Li_v / Li
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
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR, fudge_gw=fudge_gw)
        
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
    
    # inner orb sees GR&GW + SP back reaction; no LK anymore
    dLi_nat_x = (dLi_GR_x + dLi_SP_x) / Li_unit
    dLi_nat_y = (dLi_GR_y + dLi_SP_y) / Li_unit
    dLi_nat_z = (dLi_GR_z + dLi_SP_z) / Li_unit
    dei_x = dei_GR_x + dei_SP_x
    dei_y = dei_GR_y + dei_SP_y
    dei_z = dei_GR_z + dei_SP_z
    
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
            dS1_nat_x, dS1_nat_y, dS1_nat_z, \
            dS2_nat_x, dS2_nat_y, dS2_nat_z]) * t_unit
    return dy_nat_vect

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

@jit(nopython=True)
def evol_log_aL(t_nat, logy_vect, par_aL):
    loga, logL = logy_vect
    M1, M2, t_unit= par_aL
    Mt=M1+M2
    mu=M1*M2/Mt
    
    a_orb = np.exp(loga)
    L_orb = np.exp(logL)
    eff = L_orb/(mu*np.sqrt(G*Mt*a_orb))
    e_orb = np.sqrt(1.-eff**2.)
    
    G3_c5 = G**3./c**5.
    e2 = e_orb**2.
    e4 = e_orb**4.
    
    da = - (64./5.*G3_c5) * (mu*(M1+M2)**2./(a_orb**3.))\
          * (1. + 73./24.*e2 + 37./96.*e4)/(eff**7.)    
    dL = - (32./5.*G3_c5*np.sqrt(G)) * mu**2.*(M1+M2)**2.5 / (a_orb**3.5) \
          * (1.+0.875*e2)/(eff**4.)
    
    dlogy_vect = np.array([da/a_orb, dL/L_orb]) * t_unit
    return dlogy_vect

@jit(nopython=True)
def evol_logL_vs_loga(loga_Mt, logL_Mt, par_aL):
    M1, M2= par_aL
    Mt=M1+M2
    mu=M1*M2/Mt
    
    r_Mt = G*Mt/c**2.
    S_Mt = G*Mt**2./c
    
    a_orb = np.exp(loga_Mt)*r_Mt
    L_orb = np.exp(logL_Mt)*S_Mt
    eff = L_orb/(mu*np.sqrt(G*Mt*a_orb))
    e_orb = np.sqrt(1.-eff**2.)
    
    G3muMt2_c5a3 = (G*mu/c**2./a_orb) * (G*Mt/c**2./a_orb)**2. * c #G**3.*mu*Mt**2./c**5./a_orb**3.
    e2 = e_orb**2.
    e4 = e_orb**4.
    
    da = - (64./5.) * (G3muMt2_c5a3)\
          * (1. + 73./24.*e2 + 37./96.*e4)/(eff**7.)    
    dL = - (32./5.*G3muMt2_c5a3*np.sqrt(G)) * mu *np.sqrt(M1+M2) / np.sqrt(a_orb) \
          * (1.+0.875*e2)/(eff**4.)
    
    dLda = dL/da
    dlogL_dloga = a_orb/L_orb * dL/da
    return dlogL_dloga

@jit(nopython=True)
def get_angles(J, L, e, S, par):
    M1, M2, S1, S2, chi_eff = par
    
    qq=M2/M1
    Mt=M1+M2
    
    c_th1 = 0.5 * 1./((1.-qq)*S1) * \
        ((J**2. - L**2. - S**2.)/L  - 2.*qq*G*Mt**2./c*chi_eff/(1.+qq))
    c_th2 = 0.5 * qq/((1.-qq)*S2) * \
        (-(J**2.-L**2.-S**2.)/L + 2.*G*Mt**2./c*chi_eff/(1.+qq))
    c_th12 = (S**2. - S1**2. -S2**2.)/(2.*S1*S2)
    
    th1, th2, th12 = np.arccos(c_th1), np.arccos(c_th2), np.arccos(c_th12)
    c_dphi = (c_th12 - c_th1*c_th2)/(np.sin(th1) * np.sin(th2))
    dphi = np.arccos(c_dphi)
    return th1, th2, th12, dphi

@jit(nopython=True)
def get_dSdt(J, L, e, S, par):
    M1, M2, S1, S2, chi_eff = par
    
    qq=M2/M1
    Mt=M1+M2
    mu=M1*M2/Mt
    eta=mu/Mt
    eff = np.sqrt(1.-e**2.)
    
    th1, th2, th12, dphi = get_angles(J, L, e, S, par)
    
    dSdt = 1.5*eta**6. * eff**3. * (1-qq**2.)/qq * ((G*Mt**2./c)/L)**5.\
        * (c**3./G/Mt) * (S1*S2/S)\
        * (-1. + 0.5*qq/(1.-qq)**2.*(J**2.-L**2.-S**2.)/L**2. \
          - 2.*qq**2./(1.-qq**2.)**2.*G*Mt**2./c/L*chi_eff)\
        * np.sin(th1) * np.sin(th2) * np.sin(dphi)
    return dSdt

def find_Smp(J, L, e, par, nPt=1000):
    M1, M2, S1, S2, chi_eff = par
    qq=M2/M1
    Mt=M1+M2
    S_unit = G*Mt**2./c
    
    S_min = np.max([np.abs(J-L), np.abs(S1-S2)])
    S_max = np.min([J+L, S1+S2])
    S_vect = np.linspace(S_min, S_max, nPt)
    
    A1=np.sqrt(J**2. - (L-S_vect)**2.)
    A2=np.sqrt((L+S_vect)**2. - J**2.)
    A3=np.sqrt(S_vect**2. - (S1-S2)**2.)
    A4=np.sqrt((S1+S2)**2. - S_vect**2.)
    
    chi_vect1 = ((J**2.-L**2.-S_vect**2.)\
                *((S_vect**2.*(1.+qq)**2.)\
                  -(S1**2.-S2**2.)*(1.-qq**2.))\
                 -(1.-qq**2.)*A1*A2*A3*A4)\
            /(4.*qq*S_unit*S_vect**2.*L)

    chi_vect2 = ((J**2.-L**2.-S_vect**2.)\
                *((S_vect**2.*(1.+qq)**2.)\
                  -(S1**2.-S2**2.)*(1.-qq**2.))\
                 +(1.-qq**2.)*A1*A2*A3*A4)\
            /(4.*qq*S_unit*S_vect**2.*L)
    
    # note here is S vs chi-chi_eff!
    chi_vs_S_func1 = interp.interp1d(S_vect.squeeze(), chi_vect1.squeeze()-chi_eff)
    chi_vs_S_func2 = interp.interp1d(S_vect.squeeze(), chi_vect2.squeeze()-chi_eff)
    
    idx1 = np.argmin(chi_vect1)
    idx2 = np.argmax(chi_vect2)
    
    if np.max(chi_vect1) < chi_eff:
        Sm = opt.ridder(chi_vs_S_func2, S_min, S_vect[idx2])
        Sp = opt.ridder(chi_vs_S_func2, S_vect[idx2], S_max)
    elif np.min(chi_vect2) > chi_eff:
        Sm = opt.ridder(chi_vs_S_func1, S_min, S_vect[idx1])
        Sp = opt.ridder(chi_vs_S_func1, S_vect[idx1], S_max)
    else:
        try:
            Sm = opt.ridder(chi_vs_S_func1, S_min, S_vect[idx1])
        except ValueError:
            Sm = opt.ridder(chi_vs_S_func1, S_vect[idx1], S_max)
        try:
            Sp = opt.ridder(chi_vs_S_func2, S_vect[idx2], S_max)
        except ValueError:
            Sp = opt.ridder(chi_vs_S_func2, S_min, S_vect[idx2])
            
    if Sm>Sp:
        Sm, Sp = Sp, Sm
    return Sm, Sp

def find_S_chi_contour(J, L, e, par, nPt=1000):
    M1, M2, S1, S2, chi_eff = par
    qq=M2/M1
    Mt=M1+M2
    S_unit = G*Mt**2./c
    
    S_min = np.max([np.abs(J-L), np.abs(S1-S2)])
    S_max = np.min([J+L, S1+S2])
    S_vect = np.linspace(S_min, S_max, nPt)
    
    A1=np.sqrt(J**2. - (L-S_vect)**2.)
    A2=np.sqrt((L+S_vect)**2. - J**2.)
    A3=np.sqrt(S_vect**2. - (S1-S2)**2.)
    A4=np.sqrt((S1+S2)**2. - S_vect**2.)
    
    chi_vect1 = ((J**2.-L**2.-S_vect**2.)\
                *((S_vect**2.*(1.+qq)**2.)\
                  -(S1**2.-S2**2.)*(1.-qq**2.))\
                 -(1.-qq**2.)*A1*A2*A3*A4)\
            /(4.*qq*S_unit*S_vect**2.*L)

    chi_vect2 = ((J**2.-L**2.-S_vect**2.)\
                *((S_vect**2.*(1.+qq)**2.)\
                  -(S1**2.-S2**2.)*(1.-qq**2.))\
                 +(1.-qq**2.)*A1*A2*A3*A4)\
            /(4.*qq*S_unit*S_vect**2.*L)
    
    return S_vect, chi_vect1, chi_vect2
    
def get_tau_pre(J, L, e, par, nPt=1000):
    M1, M2, S1, S2, chi_eff = par

    Sm, Sp = find_Smp(J, L, e, par, nPt)
    S_vect = np.linspace(Sm, Sp, nPt)
    dSdt_vect = get_dSdt(J, L, e, S_vect, par)
    
    idx = np.isfinite(dSdt_vect)
    tau_pre = 2.*integ.trapz(1./np.abs(dSdt_vect[idx]), S_vect[idx])
    return tau_pre    

def evol_J_avg(L_nat, J_nat, e_vs_L_func, par, nPt=8000):
    M1, M2, S1, S2, chi_eff = par
    
    Mt=M1+M2
    S_Mt = G*Mt**2./c
    
    L=L_nat * S_Mt
    J=J_nat * S_Mt
    e_orb = e_vs_L_func(L)
    
    Sm, Sp = find_Smp(J, L, e_orb, par, nPt=nPt)
    S_vect = np.linspace(Sm, Sp, nPt)
    dSdt_vect = np.abs(get_dSdt(J, L, e_orb, S_vect, par))
    
    c_th_L = (J**2.+L**2.-S_vect**2.)/(2*J*L)
    tau = 2.*integ.trapz(S_Mt/dSdt_vect, S_vect/S_Mt)
    dJdL = 2./tau * integ.trapz(S_Mt*c_th_L/dSdt_vect, S_vect/S_Mt)
    return dJdL


def evol_J_avg_backup(L_nat, J_nat, e_vs_L_func, par, nPt=100):
    M1, M2, S1, S2, chi_eff = par
    
    Mt=M1+M2
    mu=M1*M2/Mt
    eta = M1*M2/Mt**2.
    
    r_Mt = G*Mt/c**2.
    S_Mt = G*Mt**2./c
    
    L=L_nat * S_Mt
    J=J_nat * S_Mt
    
    e_orb = e_vs_L_func(L)
    a_orb = L**2./(G*mu**2.*Mt*(1.-e_orb**2.))
    
    t_gw = get_inst_t_gw_from_a_orb(M1, M2, a_orb, e_orb)
    
    Sm, Sp = find_Smp(J, L, e_orb, par, nPt=nPt)
    S_vect = np.linspace(Sm, Sp, nPt)
    dSdt_vect = np.abs(get_dSdt(J, L, e_orb, S_vect, par))
    
    inv_t_pre = np.abs(dSdt_vect/(S_vect+1.e-9*np.ones(nPt)*S_Mt))
    if np.min(inv_t_pre)<(1./t_gw):
        idx = np.argmin(inv_t_pre)
        S = S_vect[idx]
        c_th_L = (J**2.+L**2.-S**2.)/(2*J*L)
        dJdL = c_th_L
    else:
        c_th_L = (J**2.+L**2.-S_vect**2.)/(2*J*L)
        tau = 2.*integ.trapz(S_Mt/dSdt_vect, S_vect/S_Mt)
        dJdL = 2./tau * integ.trapz(S_Mt*c_th_L/dSdt_vect, S_vect/S_Mt)
    return dJdL

