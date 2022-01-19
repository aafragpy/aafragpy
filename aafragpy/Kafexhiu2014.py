#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:20:43 2020

@author: sergeykoldobskiy
"""
import numpy as np


T_p_th = 0.2797
m_pi = 0.134977
m_p = 0.938272

def s_total_inelastic (T_p):
    '''
    Returns total inelastic cross_section as a function of proton kinetic energy

    Parameters
    ----------
    T_p : proton kinetic energy in GeV (laboratory frame)

    Returns
    -------
    inelastic cross-section in mb

    '''
    sigma = (30.7-0.96*np.log(T_p/T_p_th) + \
             0.18*np.power(np.log(T_p/T_p_th),2)) * \
        np.power(1-np.power((T_p_th/T_p),1.9),3)
    return sigma


def s_pi_zero (T_p):
    '''
    Returns pi0 production cross-section

    Parameters
    ----------
    T_p : proton energy, GeV

    Returns
    -------
    pi_0 production cross-section in mb

    '''
    s_0 = 7.66e-3
    M_res = 1.1883
    G_res = 0.2264
    # eta = P_pi/m_pi
    s = 2*m_p*(T_p+2*m_p)
    sqrt_s = np.sqrt(s)
    gamma = np.sqrt(M_res*M_res*(M_res*M_res+G_res*G_res))
    K = np.sqrt(8)*M_res*G_res*gamma/ (np.pi*np.sqrt(M_res*M_res+gamma))
    F_BW = (m_p*K) / ( np.power( np.power(sqrt_s-m_p,2) - M_res*M_res ,2)+M_res*M_res*G_res*G_res)
    eta = np.sqrt(np.power((s-m_pi*m_pi-4*m_p*m_p),2)-16*m_pi*m_pi*m_p*m_p) / (2*m_pi*sqrt_s)
    
    sigma_pi0 = s_0 * np.power (eta,1.95) * (1+eta+np.power(eta,5)) * np.power(F_BW,1.86)

    return sigma_pi0


def s_2pi_zero (T_p):
    '''
    Returns 2 * pi production cross-section
    Valid for 0.56 <T_p< 2 GeV
    
    Parameters
    ----------
    T_p : proton energy, GeV

    Returns
    -------
    2 * pi production cross-section in mb

    '''
    return np.where(T_p<0.56,0,5.7 / (1+np.exp(-9.3*(T_p-1.4))))

    
def p0_multiplicity (T_p,model):
    '''
    Returns multiplicity of pion production

    Parameters
    ----------
    T_p : proton kinetic energy
    model : hadronic model: GEANT, PYTHIA, SIBYLL, QGSJET

    Returns
    -------
    float
        Multiplicity of pion production

    '''
    
    if model == 'GEANT':
        T_p_lim,a_1,a_2,a_3,a_4,a_5 = 5, 0.728, 0.596, 0.491, 0.2503, 0.117
    elif model == 'PYTHIA':
        T_p_lim,a_1,a_2,a_3,a_4,a_5 = 50,0.652, 0.0016, 0.488, 0.1928, 0.483
    elif model == 'SIBYLL':
        T_p_lim,a_1,a_2,a_3,a_4,a_5 = 100,5.436, 0.254, 0.072, 0.075, 0.166
    elif model == 'QGSJET':
        T_p_lim,a_1,a_2,a_3,a_4,a_5 = 100, 0.908, 0.0009, 6.089, 0.176, 0.448   
    elif model == 'QGSJET-II-04m':
        T_p_lim,a_1,a_2,a_3,a_4,a_5 = 5, 0.908, 0.0009, 6.089, 0.176, 0.448 
         
    if T_p >= T_p_lim:
        psi_p = (T_p-3)/m_p
        return a_1*np.power(psi_p,a_4)*(1+np.exp(-a_2*np.power(psi_p,a_5))) * (1-np.exp(-a_3*np.power(psi_p,1/4)))
    if (T_p >= 1) and (T_p < 5):
        Q_p=(T_p-T_p_th)/m_p
        return -6e-3+0.237*Q_p-0.023*Q_p*Q_p      
    else:
        return 0


def sigma_pi0 (T_p,model):
    if T_p_th < T_p and T_p < 2:
        return s_pi_zero(T_p) + s_2pi_zero(T_p)
    if model =='GEANT':
        T_p_trans = 1e5
    if model =='PYTHIA':
        T_p_trans = 50
    if model =='SIBYLL':
        T_p_trans = 100
    if model =='QGSJET':
        T_p_trans = 100
    if model =='QGSJET-II-04m':
        T_p_trans = T_p_th
    if T_p>=2 and T_p<=T_p_trans:
        return s_total_inelastic(T_p)*p0_multiplicity(T_p,'GEANT')
    if T_p>T_p_trans:
        return s_total_inelastic(T_p)*p0_multiplicity(T_p,model)
        
        
def F_gamma_Kafexhiu2014 (T_p,E_g,model):
    if T_p < T_p_th:
        return 0
    b_0 = 5.9
    theta_p = T_p/m_p    
    def k(T_p):
        return 3.29 - 1/5*np.power(theta_p,-3/2)
    q = (T_p - 1)/m_p
    def mu(T_p):
        return 5/4*np.power(q,5/4)*np.exp(-5/4*q)
    
    low_energy_flag = 0
    
    if model == 'PYTHIA' and T_p>50:
        Lambda,alpha,beta,gamma = 3.5, 0.5, 4, 1 
        b_1,b_2,b_3 = 9.06, 0.3795, 0.01105
    elif model == 'SIBYLL' and T_p>100:
        Lambda,alpha,beta,gamma = 3.55, 0.5, 3.6, 1 
        b_1,b_2,b_3 = 10.77, 0.412, 0.01264
    elif model == 'QGSJET' and T_p>100:
        Lambda,alpha,beta,gamma = 3.55, 0.5, 4.5, 1 
        b_1,b_2,b_3 = 13.16, 0.4419, 0.01439
    elif model == 'QGSJET-II-04m' and T_p>T_p_th:
        Lambda,alpha,beta,gamma = 3.55, 0.5, 4.5, 1 
        b_1,b_2,b_3 = 13.16, 0.4419, 0.01439
    else:
        low_energy_flag = 1
    
    if model == 'GEANT' or low_energy_flag==1:
        if T_p_th<=T_p and T_p<1:
            Lambda,alpha,beta,gamma = 0.0, 1, k(T_p), 0
        if 1<=T_p and T_p<=4:
            Lambda,alpha,beta,gamma = 3, 1, mu(T_p)+2.45, mu(T_p)+1.45  
        if 4<T_p and T_p<=20:
            Lambda,alpha,beta,gamma = 3, 1, 3/2*mu(T_p)+4.95, mu(T_p)+1.50 
        if 20<T_p and T_p<=100:
            Lambda,alpha,beta,gamma = 3, 0.5, 4.2, 1 
        if T_p>100:
            Lambda,alpha,beta,gamma = 3, 0.5, 4.9, 1 
            
        if 1<=T_p and T_p<5: 
            b_1,b_2,b_3 = 9.53, 0.52, 0.054
        if T_p>=5: 
            b_1,b_2,b_3 = 9.13, 0.35, 9.7e-3
    
    s = 2*m_p*(T_p+2*m_p)
    sqrt_s = np.sqrt(s)
    E_pi_CM = (s-4*m_p*m_p +m_pi*m_pi)/2/sqrt_s
    P_pi_CM = np.sqrt(E_pi_CM*E_pi_CM - m_pi*m_pi)
    gamma_CM = (T_p+2*m_p)/sqrt_s
    beta_CM = np.sqrt(1-1/(gamma_CM*gamma_CM))
    E_pi_max_LAB = gamma_CM *(E_pi_CM+P_pi_CM*beta_CM)
    E_pi_max = E_pi_max_LAB
    gamma_pi_LAB = E_pi_max_LAB/m_pi
    beta_pi_LAB = np.sqrt(1-1/(gamma_pi_LAB*gamma_pi_LAB))
    E_g_max = m_pi/2 * gamma_pi_LAB * (1+beta_pi_LAB)
    Y_g = E_g+m_pi*m_pi/4/E_g
    Y_g_max = E_g_max + m_pi*m_pi/4/E_g_max
    X_g = (Y_g - m_pi) / ((Y_g_max - m_pi))
    
    
    def A_max(T_p):
        if T_p>T_p_th and T_p<1:
            return b_0*sigma_pi0(T_p,model)/E_pi_max
        elif T_p>=1:
            return b_1*np.power(theta_p,-b_2)*np.exp(b_3*np.power(np.log(theta_p),2))*sigma_pi0(T_p,model)/m_p
        
    def F(T_p,E_g):
        C = Lambda * m_pi / Y_g_max
        func = np.power(1-np.power(X_g,alpha),beta ) / np.power((1+X_g/C), gamma )
        func = np.where(E_g<=E_g_max,func,np.nan)
        return func
    
    if T_p <=1e6:
        return A_max(T_p) * F(T_p,E_g)   
    else:
        return 0
    