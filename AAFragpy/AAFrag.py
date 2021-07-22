#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:44:49 2020

@author: sergeykoldobskiy
"""
__version__ = '0.8.44'
__author__ = 'Sergey Koldobskiy'

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in")
warnings.filterwarnings("ignore", message="invalid value encountered in")
warnings.filterwarnings("ignore", message='overflow encountered in exp')

m_p = 0.9385

###############################################################################
###############################################################################


def E_trans(energy):
    """Return str with formatted energy value."""
    power = np.log10(energy)
    power_SI = power // 3
    SI = ['eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV', 'EeV']
    try:
        en = SI[int(power_SI)]
    except IndexError:
        return str(energy)+' eV'
    # print(power, power_SI, en)
    return str(int(np.round(energy/10**(power_SI*3), 2)))+' '+en

###############################################################################
###############################################################################


def interpolate_sigma(T_primary, data, le_flag, T_secondary=None):
    """Return interpolated data.

    Parameters
    ----------
        T_primary (float): Primary energy, GeV.
        data (numpy ndarray): Tabulated cross-section data.
        le_flag (int): Flag for low-energy data.
        T_secondary (list), optional
            Binning for secondaries, GeV. The default is 'data' binning.

    Returns
    -------
    temp (numpy 2D ndarray):
        Vector of secondary energy and the vector of corresponding differential
        cross-section.
    """
    # if binning is not given as input, use default one
    if T_secondary is None:
        T_sec = np.unique(data[:, 1])
        def_bin_flag = 1
    else:
        if type(T_secondary) is not np.ndarray:
            T_secondary = np.array(T_secondary)
        T_sec = T_secondary * 1e9
        def_bin_flag = 0

    log_T_i = np.log10(T_primary)
    log_data = np.log10(data)
    log_T_sec = np.log10(T_sec)

    uniq_log_T_i = np.unique(log_data[:, 0])
    uniq_T_i = np.unique(data[:, 0])
    if le_flag:
        u = (T_primary - uniq_T_i)
        idxl = np.abs(u).argsort(axis=0)[:2]
    else:
        u = (log_T_i - uniq_log_T_i)
        idxl = np.abs(u).argsort(axis=0)[:2]

    # interpolation is not needed
    if (abs(log_T_i-uniq_log_T_i[idxl[0]]) <= np.log10(1.01)
            and def_bin_flag == 1):
        # print('No interploation is needed, return tabulated data')
        temp = data[data[:, 0] == uniq_T_i[idxl[0]]][:, [1, 2]].T
        temp[0] = temp[0]/1e9
        temp[1, 0] = 0
        return temp

    cl1 = abs((log_T_i - uniq_log_T_i[idxl[0]])/(uniq_log_T_i[idxl[1]] -
                                                 uniq_log_T_i[idxl[0]]))
    cl2 = abs((log_T_i - uniq_log_T_i[idxl[1]])/(uniq_log_T_i[idxl[1]] -
                                                 uniq_log_T_i[idxl[0]]))

    si1 = log_data[np.abs(log_data[:, 0] - uniq_log_T_i[idxl[0]]) < 1e-6]
    si2 = log_data[np.abs(log_data[:, 0] - uniq_log_T_i[idxl[1]]) < 1e-6]

    a1 = si1[si1[:, 2] != -np.inf][1:, 1:]
    a2 = si2[si2[:, 2] != -np.inf][1:, 1:]

    # exception for zero matrix interpolation
    try:
        min_a1_x, max_a1_x = min(a1[:, 0]), max(a1[:, 0])
        min_a2_x, max_a2_x = min(a2[:, 0]), max(a2[:, 0])
    except ValueError:
        if def_bin_flag == 1:
            temp = data[data[:, 0] == uniq_T_i[idxl[0]]][:, [1, 2]].T
            temp[0] = temp[0]/1e9
            return temp
        if def_bin_flag == 0:
            temp = np.vstack([T_sec, np.zeros(len(T_sec))])
            return temp

    sigma_final = np.zeros(log_T_sec.shape)
    sigma_final[sigma_final == 0] = -np.inf

    filter_energies = (log_T_sec > np.min([min_a1_x, min_a2_x])) *\
        (log_T_sec < np.max([max_a1_x, max_a2_x])) * (log_T_sec < log_T_i)
    fit_energies = log_T_sec[filter_energies]
    fit_bins = np.where(filter_energies)

    new_a1_x = np.linspace(min_a1_x, max_a1_x, 1000)
    new_a2_x = np.linspace(min_a2_x, max_a2_x, 1000)

    new_a1_y = np.interp(new_a1_x, a1[:, 0], a1[:, 1])
    new_a2_y = np.interp(new_a2_x, a2[:, 0], a2[:, 1])

    midx = cl2*new_a1_x+cl1*new_a2_x
    midy = cl2*new_a1_y+cl1*new_a2_y

    sigma_final[fit_bins] = np.interp(fit_energies, midx, midy)

    temp = np.array((T_sec, np.power(10, sigma_final)))
    temp[0] = temp[0]/1e9
    # if def_bin_flag == 0:
    #     temp = temp[:, 0, :]

    # check that last 15 values are not rise or equal
    if (temp[:, np.argmax(temp[1]):][1] != 0).sum() > 15:
        last15_v = temp[1][np.where(temp[1] != 0)][-15:]
        last14_i = np.array(np.where(temp[1] != 0))[:, -14:]
        check14_v = np.where(last15_v[-14:] < last15_v[:14], last15_v[:14], 0)
        temp[1, last14_i] = check14_v
    else:
        print('Differential cross-section values can be \
              wrongly calculated close to kinematic threshold. \
              Please increse the number of bins.')

    return temp

###############################################################################
###############################################################################


def open_data_files(secondary, primary_target):
    """Open AAFrag data files."""
    import os
    import inspect
    AAFrag_path = (os.path.dirname(inspect.getfile(open_data_files)))
    if secondary == 'gam':
        data_col = 2
    elif secondary == 'el':
        data_col = 2
    elif secondary == 'posi':
        secondary = 'el'
        data_col = 3
    elif secondary == 'nu_e':
        secondary = 'nu'
        data_col = 2
    elif secondary == 'anu_e':
        secondary = 'nu'
        data_col = 3
    elif secondary == 'nu_mu':
        secondary = 'nu'
        data_col = 4
    elif secondary == 'anu_mu':
        secondary = 'nu'
        data_col = 5
    elif secondary == 'p':
        secondary = 'pap'
        data_col = 2
    elif secondary == 'ap':
        secondary = 'pap'
        data_col = 3
    elif secondary == 'n':
        secondary = 'nan'
        data_col = 2
    elif secondary == 'an':
        secondary = 'nan'
        data_col = 3
    else:
        return print('Unknown product. Check your input please!')

    name = secondary+'_'+primary_target.split('-')[0]+'_' +\
        primary_target.split('-')[1]

    try:
        # data_HE = pd.read_csv(AAFrag_path+'/Tables/'+name+'_04',
        #                       delim_whitespace=True, header=None)
        # data_HE = np.array(data_HE)
        
        data_HE = np.genfromtxt(AAFrag_path+'/Tables/'+name+'_04')
        data_HE = data_HE[:, [0, 1, data_col]]
        data_LE = 0
    except OSError:
        return print('There is no data for this combination of primary\
                     \nand target. Check your input please!')

    E_th_b = float(data_HE[0, 0])
    E_th_t = float(data_HE[-1:, 0])
    E_th_c = 0

    try:
        # data_LE = pd.read_csv(AAFrag_path+'/Tables/'+name+'_04L',
        #                       delim_whitespace=True, header=None)
        # data_LE = np.array(data_LE)

        data_LE = np.genfromtxt(AAFrag_path+'/Tables/'+name+'_04L')
        data_LE = data_LE[:, [0, 1, data_col]]
        
        E_th_b = float(data_LE[0, 0])
        E_th_c = float(data_LE[-1:, 0])
    except OSError:
        pass

    return data_HE, data_LE, E_th_b, E_th_c, E_th_t

###############################################################################
###############################################################################


def get_cs_value(secondary, primary_target, E_primary,
                        T_secondaries=None):
    """
    Return single differential cross-section value.

    Parameters
    ----------
    secondary (str): Partcile-product of nucleon-nucleon interaction.
        Allowed inputs are: gam, posi, el, nu_e, anu_e, mu_mu, amu_mi
    primary_target (str): Primary/target combination.
    T_primary (float): Kinetic energy of primary particle in GeV.
    T_secondaries (list or tuple or numpy.ndarray): optional
        Array for  energy bins for the secondary particles.
        Default (tabulated) binning is used if input is empty.

    Returns
    -------
    2d numpy array (secondary energy, secondary differental cross-section)

    """
    E_primary = E_primary * 1e9
    try:
        data_HE, data_LE, E_th_b, E_th_c, E_th_t = open_data_files(secondary,
                                                               primary_target)
    except TypeError:
        return 
    
    if E_th_b/E_primary < 1.001 and E_primary/E_th_t < 1.001:
        le_flag = 1
        if E_primary - E_th_c >= 9e-3:
            le_flag = 0
        if (T_secondaries is None):
            data = interpolate_sigma(E_primary, data_HE, le_flag)
        else:
            data = interpolate_sigma(E_primary, data_HE, le_flag,
                                     T_secondaries)
        if le_flag == 1:
            if (T_secondaries is None):
                data = interpolate_sigma(E_primary, data_LE, le_flag)
            else:
                data = interpolate_sigma(E_primary, data_LE, le_flag,
                                         T_secondaries)
        data[1] = data[1]/data[0]
    else:
        return print('Primary kinetic energy '+E_trans(E_primary) +
                     ' is not in range: '+E_trans(E_th_b)+' -- ' +
                     E_trans(E_th_t) +
                     ' avaliable for primary/target combination: ' +
                     primary_target)

    return data

###############################################################################
###############################################################################


def get_cs_matrix(secondary, primary_target, E_primaries=None,
                  T_secondaries=None):
    """
    Calculate mutliple cross-section values.

    Return matrix vectors of primary energy, secondary energy
    and the matrix of differential cross-section for given
    combination of energy vectors.

    Parameters
    ----------
    secondary (str): Secondary partcile of nucleon-nucleon interaction.
    primary_target (str): Primary/target combination.
    T_primaries (list or tuple or numpy.ndarray): optional
        Array for energy bins for the projectile particles.
        The default is taken from tables.
    T_secondaries (list or tuple or numpy.ndarray): optional
        Array for energy bins for the secondary particles.
        The default is taken from tables.

    Returns
    -------
    (numpy ndarray 1D)
        Vector of primary energy.
    (numpy ndarray 1D)
        Vector of secondary energy.
    (numpy ndarray 2D)
        Matrix of differential cross-section for given combination of vectors.
    """
    try:
        data_HE, data_LE, E_th_b, E_th_c, E_th_t = open_data_files(secondary,
                                                               primary_target)
    except TypeError:
        return 
    
    if (E_primaries is None) and (T_secondaries is None):

        energy_primary = np.unique(data_HE[:, 0])/1e9
        len_en_primary = len(energy_primary)
        energy_secondary = np.unique(data_HE[:, 1])/1e9
        len_en_secondary = len(energy_secondary)
        cs_matrix = np.reshape(data_HE[:, 2],
                               [len_en_primary, len_en_secondary])

        if not np.isscalar(data_LE):
            energy_primary_LE = np.unique(data_LE[:, 0])/1e9
            len_en_primary_LE = len(energy_primary_LE)
            len_en_secondary_LE = len(np.unique(data_LE[:, 1]))
            cs_matrix_LE = np.reshape(data_LE[:, 2], [len_en_primary_LE,
                                                      len_en_secondary_LE])

            cs_matrix = np.vstack([cs_matrix_LE[:-1], cs_matrix])
            energy_primary = np.hstack([energy_primary_LE[:-1],
                                        energy_primary])

        cs_matrix[:, 0] = 0
        cs_matrix = cs_matrix/energy_secondary

    else:
        
        if (E_primaries is None):
            E_primaries = np.unique(data_HE[:, 0])/1e9
            if not np.isscalar(data_LE):
                energy_primary_LE = np.unique(data_LE[:, 0])/1e9
                E_primaries = np.hstack([energy_primary_LE[:-1], E_primaries])

        if type(E_primaries) is not np.ndarray:
            E_primaries = np.array(E_primaries)
        E_primaries = E_primaries * 1e9


        E_max = E_primaries.max()
        E_min = E_primaries.min()
        if E_th_b/E_min > 1.001 or E_max/E_th_t > 1.001:
            print('check')
            return print('Primary kinetic energy range is not in range: ' +
                          E_trans(E_th_b)+' -- '+E_trans(E_th_t) +
                          ' avaliable for primary/target combination: ' +
                          primary_target)
            not_in_range = 1
        else:
            not_in_range = 0
            c = 0
            for E_primary in E_primaries:
    
                if E_th_b/E_primary < 1.001 and E_primary/E_th_t < 1.001:
                    le_flag = 1
                    if E_primary - E_th_c >= 9e-3:
                        le_flag = 0
    
                    if (T_secondaries is None):
                        if le_flag == 1:
                            new_data = interpolate_sigma(E_primary,
                                                         data_LE, le_flag)
                        else:
                            new_data = interpolate_sigma(E_primary,
                                                         data_HE, le_flag)
                    else:
                        if le_flag == 1:
                            new_data = interpolate_sigma(E_primary, data_LE,
                                                         le_flag, T_secondaries)
                        else:
                            new_data = interpolate_sigma(E_primary, data_HE,
                                                         le_flag, T_secondaries)
    
                    if c == 0:
                        cs_matrix = new_data[1]
                        energy_primary = E_primary/1e9
                        energy_secondary = new_data[0]
                    else:
                        cs_matrix = np.vstack([cs_matrix, new_data[1]])
                        energy_primary = np.vstack([energy_primary, E_primary/1e9])
                    c += 1

        if not_in_range == 0:
            cs_matrix = cs_matrix / energy_secondary

    return energy_primary, energy_secondary, cs_matrix

###############################################################################
###############################################################################


def get_cs_matrix_Kafexhiu2014(E_primaries=None,
                            T_secondaries=None):
    
    from Kafexhiu2014 import F_gamma_Kafexhiu2014
    csf = np.vectorize(F_gamma_Kafexhiu2014)
    
    if (E_primaries is None) or (T_secondaries is None):
        return print('Error: please provide your binning for proton \
                     energies and secondary particles.')
    else:
        if type(E_primaries) is not np.ndarray:
            E_primaries = np.array(E_primaries)
        if type(T_secondaries) is not np.ndarray:
            T_secondaries = np.array(T_secondaries)

        cs_matrix = np.zeros([len(E_primaries), len(T_secondaries)])

        for i, E_p in enumerate(E_primaries):
            cs_matrix[i] = csf(E_p-0.9385, T_secondaries, 'GEANT')

        return E_primaries, T_secondaries, cs_matrix


###############################################################################
###############################################################################


def get_cs_matrix_Kamae2006(secondary, E_primaries=None,
                            T_secondaries=None, diffractive=True):
    """
    Return mutiple cross-section values (Kamae et al. 2006).

    Return matrix vectors of primary energy, secondary energy and
        the matrix of differential cross-section for given
        combination of energy vectors.
    Based on Kamae et al. 2006
    Calculations are given for p-p interactions
        and for gamma and lepton production only.
    Works good in low energies,
        but should be substituted by newer codes in high energies.
    ----------
    secondary (str): Secondary partcile of nucleon-nucleon interaction.
    T_primaries (list or tuple or numpy.ndarray): optional
        Array for energy bins for the projectile particles.
    T_secondaries (list or tuple or numpy.ndarray): optional
        Array for energy bins for the secondary particles.

    Returns
    -------
    (numpy ndarray 1D)
        Vector of primary energy.
    (numpy ndarray 1D)
        Vector of secondary energy.
    (numpy ndarray 2D)
        Matrix of differential cross-section for given combination of vectors.
    """
    if secondary == 'gam':
        from Kamae2006 import dXSdE_gamma_Kamae2006
        csf = np.vectorize(dXSdE_gamma_Kamae2006)
    elif secondary == 'el':
        from Kamae2006 import dXSdE_elec_Kamae2006
        csf = np.vectorize(dXSdE_elec_Kamae2006)
    elif secondary == 'posi':
        from Kamae2006 import dXSdE_posi_Kamae2006
        csf = np.vectorize(dXSdE_posi_Kamae2006)
    elif secondary == 'nu_e':
        from Kamae2006 import dXSdE_elec_nu_Kamae2006
        csf = np.vectorize(dXSdE_elec_nu_Kamae2006)
    elif secondary == 'anu_e':
        from Kamae2006 import dXSdE_elec_anti_nu_Kamae2006
        csf = np.vectorize(dXSdE_elec_anti_nu_Kamae2006)
    elif secondary == 'nu_mu':
        from Kamae2006 import dXSdE_mu_nu_Kamae2006
        csf = np.vectorize(dXSdE_mu_nu_Kamae2006)
    elif secondary == 'anu_mu':
        from Kamae2006 import dXSdE_mu_anti_nu_Kamae2006
        csf = np.vectorize(dXSdE_mu_anti_nu_Kamae2006)
    else:
        def csf(T_secondaries, T_primaries):
            return np.zeros(len(T_secondaries))

    if (E_primaries is None) or (T_secondaries is None):
        return print('Error: please provide your binning for proton \
                     energies and secondary particles.')
    else:
        if type(E_primaries) is not np.ndarray:
            E_primaries = np.array(E_primaries)
        if type(T_secondaries) is not np.ndarray:
            T_secondaries = np.array(T_secondaries)

        cs_matrix = np.zeros([len(E_primaries), len(T_secondaries)])

        for i, E_p in enumerate(E_primaries):
            if E_p < 512e3:
                cs_matrix[i] = csf(E_p-0.9385, T_secondaries, diffractive)

        return E_primaries, T_secondaries, cs_matrix

###############################################################################
###############################################################################


def power_law(E, J0, gamma):
    """Return simple power-law spectrum."""
    
    p = np.sqrt(E**2-m_p**2)
    dn_dp = p**-gamma
    dE_dp = p/E
    dn_dE = dn_dp/dE_dp
    
    return J0 * dn_dE

###############################################################################
###############################################################################


def power_law_exp_decay(E, J0, gamma, p_cut):
    """Return simple power-law spectrum with exp decay."""
    
        
    p = np.sqrt(E**2-m_p**2)
    dn_dp = p**-gamma
    dE_dp = p/E
    dn_dE = dn_dp/dE_dp*np.exp(-p/p_cut)
    
    return J0 * dn_dE

###############################################################################
###############################################################################

###############################################################################
###############################################################################


def get_spectrum(energy_primary, energy_secondary, cs_matrix, prim_spectrum):
    """
    Calculate spectrum of secondary particles.

    Generates differential secondary particle spectrum given
        cross-seection matrix and primary spectrum.

    Parameters
    ----------
    energy_primary (numpy ndarray): Vector of primary energies, GeV
    energy_secondary (numpy ndarray): Vector of secondary energies, GeV
    cs_matrix (numpy ndarray): Matrix of cross-section
    nucl_spectrum (numpy ndarray): Primary spectrum

    Returns
    -------
    Differential spectrum of secondary particles (2d numpy array)

    """
    def integral(Y1, Y2, E1, E2):
        INT = (Y2*E2-Y1*E1)/((np.log(Y2/Y1)/np.log(E2/E1)+1))
        return INT
    if len(prim_spectrum.shape) == 2:
        prim_spectrum = prim_spectrum[:, 0]
    E1 = energy_primary[:-1]
    E2 = energy_primary[1:]
    Y1 = (cs_matrix[:-1].T * prim_spectrum[:-1]).T
    Y2 = (cs_matrix[1:].T * prim_spectrum[1:]).T
    int_value = np.zeros(cs_matrix.shape[1])
    for i, v in enumerate(E1):
        int_temp = (integral(Y1[i], Y2[i], E1[i], E2[i]))
        int_temp[np.isnan(int_temp)] = 0
        int_value += int_temp

    return energy_secondary, int_value
