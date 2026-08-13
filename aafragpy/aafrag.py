#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:44:49 2020

@author: sergeykoldobskiy
"""

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in")
warnings.filterwarnings("ignore", message="invalid value encountered in")
warnings.filterwarnings("ignore", message='overflow encountered in exp')

m_p = 0.938272

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
    return str((np.round(energy/10**(power_SI*3), 1)))+' '+en

###############################################################################
###############################################################################


def interpolate_sigma(E_primary, data, le_flag, E_secondary=None):
    """Return interpolated data.

    Parameters
    ----------
        E_primary (float): Primary energy, GeV.
        data (numpy ndarray): Tabulated cross-section data.
        le_flag (int): Flag for low-energy data.
        E_secondary (list), optional
            Binning for secondaries, GeV. The default is 'data' binning.

    Returns
    -------
    temp (numpy 2D ndarray):
        Vector of secondary energy and the vector of the corresponding
        differential cross-section.
    """
    # if binning is not given as input, use default one
    if E_secondary is None:
        E_sec = np.unique(data[:, 1])
        def_bin_flag = 1
    else:
        if type(E_secondary) is not np.ndarray:
            E_secondary = np.array(E_secondary)
        E_sec = E_secondary * 1e9
        def_bin_flag = 0

    log_E_i = np.log10(E_primary)
    log_E_sec = np.log10(E_sec)

    uniq_E_i = np.unique(data[:, 0])

    if le_flag:
        u = (E_primary - uniq_E_i)
        idxl = np.abs(u).argsort(axis=0)[:2]
        uniq_log_E_i = np.log10(uniq_E_i)
    else:
        uniq_log_E_i = np.log10(uniq_E_i)
        u = (log_E_i - uniq_log_E_i)
        idxl = np.abs(u).argsort(axis=0)[:2]

    # interpolation is not needed
    if (abs(log_E_i-uniq_log_E_i[idxl[0]]) <= np.log10(1.01)
            and def_bin_flag == 1):
        # print('No interploation is needed, return tabulated data')
        temp = data[data[:, 0] == uniq_E_i[idxl[0]]][:, [1, 2]].T
        temp[0] = temp[0]/1e9
        temp[1, 0] = 0
        return temp

    cl1 = abs((log_E_i - uniq_log_E_i[idxl[0]])/(uniq_log_E_i[idxl[1]] -
                                                 uniq_log_E_i[idxl[0]]))
    cl2 = abs((log_E_i - uniq_log_E_i[idxl[1]])/(uniq_log_E_i[idxl[1]] -
                                                 uniq_log_E_i[idxl[0]]))

    # Only log10 the selected blocks to avoid full log_data creation
    si1_data = data[data[:, 0] == uniq_E_i[idxl[0]]]
    si2_data = data[data[:, 0] == uniq_E_i[idxl[1]]]
    si1 = np.log10(si1_data)
    si2 = np.log10(si2_data)
    
    #get indices of the last inf in low energies
    inf_si1 = np.where(si1[:,2][si1[:,1]<8]==-np.inf)[0][-1]
    inf_si2 = np.where(si2[:,2][si2[:,1]<8]==-np.inf)[0][-1]
    
    si1[:,2] = np.where(np.where(si1[:,2])[0]<inf_si1,-np.inf,si1[:,2])
    si2[:,2] = np.where(np.where(si2[:,2])[0]<inf_si2,-np.inf,si2[:,2])

    a1 = si1[si1[:, 2] != -np.inf][1:, 1:]
    a2 = si2[si2[:, 2] != -np.inf][1:, 1:]

    # exception for zero matrix interpolation
    try:
        min_a1_x, max_a1_x = min(a1[:, 0]), max(a1[:, 0])
        min_a2_x, max_a2_x = min(a2[:, 0]), max(a2[:, 0])
    except ValueError:
        if def_bin_flag == 1:
            temp = data[data[:, 0] == uniq_E_i[idxl[0]]][:, [1, 2]].T
            temp[0] = temp[0]/1e9
            return temp
        if def_bin_flag == 0:
            temp = np.vstack([E_sec, np.zeros(len(E_sec))])
            return temp

    sigma_final = np.zeros(log_E_sec.shape)
    sigma_final[sigma_final == 0] = -np.inf

    new_a1_x = np.linspace(min_a1_x, max_a1_x, 1000)
    new_a2_x = np.linspace(min_a2_x, max_a2_x, 1000)

    new_a1_y = np.interp(new_a1_x, a1[:, 0], a1[:, 1])
    new_a2_y = np.interp(new_a2_x, a2[:, 0], a2[:, 1])

    midx = cl2*new_a1_x+cl1*new_a2_x
    midy = cl2*new_a1_y+cl1*new_a2_y


    filter_energies = (log_E_sec > np.min([min_a1_x, min_a2_x])) *\
        (log_E_sec < np.max([max_a1_x, max_a2_x])) * (log_E_sec <= log_E_i) *\
            (log_E_sec <= max(midx)) * (log_E_sec >=min(midx))
    fiE_energies = log_E_sec[filter_energies]
    fiE_bins = np.where(filter_energies)
    
    sigma_final[fiE_bins] = np.interp(fiE_energies, midx, midy)

    temp = np.array((E_sec, np.power(10, sigma_final)))
    temp[0] = temp[0]/1e9

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
    elif secondary == 'nu_all':
        secondary = 'nu'
        data_col = 100
    else:
        return print('Unknown product. Check your input, please!')

    name = secondary+'_'+primary_target.split('-')[0]+'_' +\
        primary_target.split('-')[1]

    try:
        
        if os.path.exists(AAFrag_path+'/Tables/'+name+'_04.npz'):
            data_HE = np.load(AAFrag_path+'/Tables/'+name+'_04.npz')['data']
        else:
            data_HE = np.genfromtxt(AAFrag_path+'/Tables/'+name+'_04')

        if data_col != 100:
            data_HE = data_HE[:, [0, 1, data_col]]
        else:
            temp_nu = data_HE[:,[2,3,4,5]].sum(axis=1)
            data_HE = np.vstack([data_HE[:, [0, 1]].T,temp_nu]).T
        data_LE = 0
    except OSError:
        return print('There is no data for this combination of primary'+\
                     'and target. Check your input, please!')

    E_th_b = np.float64(data_HE[0, 0])
    E_th_t = np.float64(data_HE[-1, 0])
    E_th_c = 0

    try:

        if os.path.exists(AAFrag_path+'/Tables/'+name+'_04L.npz'):
            data_LE = np.load(AAFrag_path+'/Tables/'+name+'_04L.npz')['data']
        else:
            data_LE = np.genfromtxt(AAFrag_path+'/Tables/'+name+'_04L')

        if data_col != 100:
            data_LE = data_LE[:, [0, 1, data_col]]
        else:
            temp_nu = data_LE[:,[2,3,4,5]].sum(axis=1)
            data_LE = np.vstack([data_LE[:, [0, 1]].T,temp_nu]).T
        E_th_b = np.float64(data_LE[0, 0])
        E_th_c = np.float64(data_LE[-1, 0])
    except OSError:
        pass

    return data_HE, data_LE, E_th_b, E_th_c, E_th_t

###############################################################################
###############################################################################


def get_cs_value(secondary, primary_target, E_primaries,
                        E_secondaries=None):
    """
    Return single differential cross-section value.

    Parameters
    ----------
    secondary (str): Secondary particle produced in the nucleon-nucleon
    interaction.
        Allowed inputs are: gam, posi, el, nu_e, anu_e, mu_mu, amu_mu, nu_all
    primary_target (str): Primary/target combination.
    E_primaries (int or float): Total energy of a primary particle in GeV.
    E_secondaries (int or float or list or tuple or numpy.ndarray): optional
    Vector of the secondary particle energy (in GeV).
        Default (tabulated) binning is used if the input is empty.

    Returns
    -------
    2d numpy array (secondary differential cross-section, secondary energy)

    """
    res = get_cross_section(secondary, primary_target, E_primaries, E_secondaries)
    if res is None:
        return None
    cs_matrix, _, e_s = res
    if cs_matrix.ndim == 2:
        return np.array([cs_matrix[0], e_s])
    return np.array([cs_matrix, e_s])

###############################################################################
###############################################################################


def get_cross_section(secondary, primary_target, E_primaries=None,
                  E_secondaries=None, extrapolate='raise'):
    """
    Reconstruсt cross-section values for given values of the total energy for
    primary and secondary particle combination.

    Return the matrix of differential cross-section, vector of primary total
    energy and secondary energy.
    
    If primary and secondary energies are not set, the default binning will be used.

    Parameters
    ----------
    secondary (str): Secondary particle produced in the nucleon-nucleon
    interaction.
        Allowed inputs are: gam, posi, el, nu_e, anu_e, mu_mu, amu_mu, nu_all
    primary_target (str): Primary/target combination.
    E_primaries (int or float or list or tuple or numpy.ndarray): optional
        Vector of the primary particle energy (in GeV) of the size M.
        The default values are taken from the tables.
    E_secondaries (int or float or list or tuple or numpy.ndarray): optional
        Vector of the secondary particle energy (in GeV) of the size N.
        The default values are taken from the tables.
    extrapolate (str): Defines behavior if E_primaries is outside tabulated range.
        Allowed inputs: 'raise' (raises ValueError), 'nan' (outputs np.nan array),
        'interpolate' (linearly extrapolates log-log data from the bounds).

    Returns
    -------
    (numpy ndarray 2D)
        Matrix MxN of differential cross-section (in mb/GeV) for a given
        combination of vectors.
    (numpy ndarray 1D)
        Vector of primary total energy in GeV.
    (numpy ndarray 1D)
        Vector of secondary energy in GeV.

    """
    try:
        data_HE, data_LE, E_th_b, E_th_c, E_th_t = open_data_files(secondary,
                                                               primary_target)
    except TypeError:
        return 
    
    # primary = primary_target.split('-')[0]
    # avaliable_primaries = ['p','He','C','Al','Fe']
    # masses = [0.9385,3.7274,11.178,25.133,52.103]
    # mass = masses[avaliable_primaries==primary]
    
    if (E_primaries is None) and (E_secondaries is None):

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
        else: 
            if type(E_primaries) is not np.ndarray:
                if np.isscalar(E_primaries):
                    E_primaries = [E_primaries]
                E_primaries = np.array(E_primaries)

        E_primaries = E_primaries * 1e9


        E_max = E_primaries.max()
        E_min = E_primaries.min()

        c = 0
        for E_primary in E_primaries:
            out_of_range = (E_th_b/E_primary > 1.001 or E_primary/E_th_t > 1.001)

            if out_of_range:
                if extrapolate == 'raise':
                    raise ValueError('Primary total energy is not in range: ' +
                                  E_trans(E_th_b)+' -- '+E_trans(E_th_t) +
                                  ' avaliable for primary/target combination: ' +
                                  primary_target)
                elif extrapolate == 'nan':
                    le_flag = 0
                    if E_secondaries is None:
                        E_sec = np.unique(data_HE[:, 1])
                    else:
                        if type(E_secondaries) is not np.ndarray:
                            if np.isscalar(E_secondaries):
                                E_secondaries = [E_secondaries]
                            E_secondaries = np.array(E_secondaries)
                        E_sec = E_secondaries * 1e9
                    new_data = np.array([E_sec, np.full(len(E_sec), np.nan)])
                    new_data[0] = new_data[0]/1e9
                elif extrapolate == 'interpolate':
                    le_flag = 0
                    if (E_secondaries is None):
                        new_data = interpolate_sigma(E_primary, data_HE, le_flag)
                    else:
                        if type(E_secondaries) is not np.ndarray:
                            if np.isscalar(E_secondaries):
                                E_secondaries = [E_secondaries]
                            E_secondaries = np.array(E_secondaries)
                        new_data = interpolate_sigma(E_primary, data_HE, le_flag, E_secondaries)
                else:
                    return print('Primary total energy is not in range: ' +
                                  E_trans(E_th_b)+' -- '+E_trans(E_th_t) +
                                  ' avaliable for primary/target combination: ' +
                                  primary_target)
            else:
                le_flag = 1
                if E_primary - E_th_c >= 9e-3:
                    le_flag = 0

                if (E_secondaries is None):
                    if le_flag == 1:
                        new_data = interpolate_sigma(E_primary,
                                                     data_LE, le_flag)
                    else:
                        new_data = interpolate_sigma(E_primary,
                                                     data_HE, le_flag)
                else:
                    if type(E_secondaries) is not np.ndarray:
                        if np.isscalar(E_secondaries):
                            E_secondaries = [E_secondaries]
                        E_secondaries = np.array(E_secondaries)
                    if le_flag == 1:
                        new_data = interpolate_sigma(E_primary, data_LE,
                                                     le_flag, E_secondaries)
                    else:
                        new_data = interpolate_sigma(E_primary, data_HE,
                                                     le_flag, E_secondaries)

            if c == 0:
                cs_matrix_list = [new_data[1]]
                energy_primary_list = [E_primary/1e9]
                energy_secondary = new_data[0]
            else:
                cs_matrix_list.append(new_data[1])
                energy_primary_list.append(E_primary/1e9)
            c += 1

        cs_matrix = np.array(cs_matrix_list)
        energy_primary = np.array(energy_primary_list)

        cs_matrix = cs_matrix / energy_secondary
        
        if c == 1:
            energy_primary = np.array([energy_primary])
            cs_matrix = np.array([cs_matrix])
            return cs_matrix, (energy_primary), energy_secondary

    return cs_matrix, np.squeeze(energy_primary), energy_secondary

###############################################################################
###############################################################################


def get_cross_section_Kafexhiu2014(E_primaries, E_secondaries):
    """
    Return cross-section values (Kafexhiu et al. 2014).

    Return the matrix of the differential cross-section for a given
        combination of energy vectors, primary energy vector, secondary energy
        vector.
        
    Based on Kafexhiu et al. 2014 (GEANT parameters)
    Calculations are performed for p-p interactions
        and for gamma production only.
    Works good in low energies,
        but should be substituted by newer codes in high energies.
    ----------
    E_primaries (int or float or list or tuple or numpy.ndarray): 
        Vector of the primary proton energy (in GeV) of the size M.
    E_secondaries (int or float or list or tuple or numpy.ndarray):
        Vector of the gamma energy (in GeV) of the size N.

    Returns
    -------
    (numpy ndarray 2D)
        Matrix MxN of the differential cross-section (in mb/GeV) 
        for a given combination of vectors.
    (numpy ndarray 1D)
        Vector of primary energy in GeV.
    (numpy ndarray 1D)
        Vector of secondary energy in GeV.
    """
    from Kafexhiu2014 import F_gamma_Kafexhiu2014
    csf = np.vectorize(F_gamma_Kafexhiu2014)
    
    if (E_primaries is None) or (E_secondaries is None):
        return print('Error: please provide the energy binning for protons'+\
                     ' and secondary particles.')
    else:
        if type(E_primaries) is not np.ndarray:
            if np.isscalar(E_primaries):
                E_primaries = [E_primaries]
            E_primaries = np.array(E_primaries)
        if type(E_secondaries) is not np.ndarray:
            if np.isscalar(E_secondaries):
                E_secondaries = [E_secondaries]
            E_secondaries = np.array(E_secondaries)
        cs_matrix = np.zeros([len(E_primaries), len(E_secondaries)])

        for i, E_p in enumerate(E_primaries):
            cs_matrix[i] = csf(E_p-m_p, E_secondaries, 'GEANT')

        return cs_matrix, E_primaries, E_secondaries


###############################################################################
###############################################################################


def get_cross_section_Kamae2006(secondary, E_primaries,
                            E_secondaries, diffractive=True):
    """
    Return  cross-section values (Kamae et al. 2006).

    Return the matrix of the differential cross-section for a given
        combination of energy vectors, primary energy vector, secondary energy
        vector.
    Based on Kamae et al. 2006
    Calculations are performed for p-p interactions
        and for gamma and lepton production only.
    Works good in low energies,
        but should be substituted by newer codes in high energies.
    ----------
    secondary (str): Secondary particle of proton-proton interaction.
    E_primaries (int or float or list or tuple or numpy.ndarray): 
        Vector of the primary proton energy (in GeV) of the size M.
    E_secondaries (int or float or list or tuple or numpy.ndarray):
        Vector of the secondary particle energy (in GeV) of the size N.
    diffractive (bool): Include or exclude diffractive processes

    Returns
    -------
    (numpy ndarray 2D)
        Matrix MxN of the differential cross-section (in mb/GeV)
        for a given combination of vectors.
    (numpy ndarray 1D)
        Vector of primary energy in GeV.
    (numpy ndarray 1D)
        Vector of secondary energy in GeV.
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
    elif secondary == 'nu_all':
        from Kamae2006 import dXSdE_elec_nu_Kamae2006, dXSdE_elec_anti_nu_Kamae2006,\
        dXSdE_mu_nu_Kamae2006, dXSdE_mu_anti_nu_Kamae2006
        def nu_sum_Kamae (T_p, T_secondaries, diffractive):
            return dXSdE_elec_nu_Kamae2006(T_p, T_secondaries, diffractive) +\
            dXSdE_elec_anti_nu_Kamae2006(T_p, T_secondaries, diffractive) +\
            dXSdE_mu_nu_Kamae2006(T_p, T_secondaries, diffractive) + \
            dXSdE_mu_anti_nu_Kamae2006(T_p, T_secondaries, diffractive)
        csf = np.vectorize(nu_sum_Kamae)
    else:
        def csf(T_secondaries, T_primaries):
            return np.zeros(len(T_secondaries))

    if (E_primaries is None) or (E_secondaries is None):
        return print('Error: please provide the energy binning for protons'+\
                     ' and secondary particles.')
    else:
        if type(E_primaries) is not np.ndarray:
            if np.isscalar(E_primaries):
                E_primaries = [E_primaries]
            E_primaries = np.array(E_primaries)
        if type(E_secondaries) is not np.ndarray:
            if np.isscalar(E_secondaries):
                E_secondaries = [E_secondaries]
            E_secondaries = np.array(E_secondaries)

        cs_matrix = np.zeros([len(E_primaries), len(E_secondaries)])

        for i, E_p in enumerate(E_primaries):
            if E_p < 512e3:
                cs_matrix[i] = csf(E_p-m_p, E_secondaries, diffractive)

        return cs_matrix, E_primaries, E_secondaries


###############################################################################
###############################################################################


def get_spectrum(energy_primary, energy_secondary, cs_matrix, prim_spectrum):
    """
    Calculate the spectrum of secondary particles.

    Generates differential secondary particle spectrum for given
        secondary differential cross-section matrix and primary spectrum.

    Parameters
    ----------
    energy_primary (numpy ndarray): Vector of primary energies, GeV
    energy_secondary (numpy ndarray): Vector of secondary energies, GeV
    cs_matrix (numpy ndarray): Matrix of differential cross-section
    prim_spectrum (numpy ndarray): Primary spectrum

    Returns
    -------
    (numpy ndarray 1D)
    Differential spectrum of secondary particles

    """
    if len(prim_spectrum.shape) == 2:
        prim_spectrum = prim_spectrum[:, 0]

    E1 = energy_primary[:-1, np.newaxis]
    E2 = energy_primary[1:, np.newaxis]
    Y1 = cs_matrix[:-1] * prim_spectrum[:-1, np.newaxis]
    Y2 = cs_matrix[1:] * prim_spectrum[1:, np.newaxis]

    INT = (Y2*E2-Y1*E1)/((np.log(Y2/Y1)/np.log(E2/E1)+1))
    INT[np.isnan(INT)] = 0
    return np.sum(INT, axis=0)
