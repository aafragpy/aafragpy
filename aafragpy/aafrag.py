#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:44:49 2020

@author: sergeykoldobskiy
"""

import os
import warnings
from functools import lru_cache
import numpy as np

# Suppress expected mathematical/numerical warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in")
warnings.filterwarnings("ignore", message="invalid value encountered in")
warnings.filterwarnings("ignore", message="overflow encountered in exp")

m_p = 0.938272
AAFrag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tables')


# Universal default secondary energy grid (17 decades, 100 pts/dec, exact 0.01 dex step)
# Implemented for backward compatability with the previous code
MASTER_E_SECONDARIES = np.logspace(-5, 12, 1701)
###############################################################################
###############################################################################


def E_trans(energy):
    """
    Return formatted string with energy value and SI prefix.

    Parameters
    ----------
    energy : float
        Energy in eV.

    Returns
    -------
    str
        Formatted energy string (e.g., '10.0 GeV').
    """
    power = np.log10(energy)
    power_SI = power // 3
    SI = ['eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV', 'EeV']
    try:
        en = SI[int(power_SI)]
    except IndexError:
        return f"{energy} eV"
    return f"{np.round(energy / 10**(power_SI * 3), 1)} {en}"


###############################################################################
###############################################################################


@lru_cache(maxsize=64)
def open_data_files(secondary, primary_target):
    """
    Open AAFrag data natively from consolidated 3D .npz matrix archives.

    Parameters
    ----------
    secondary : str
        Secondary particle produced in the interaction.
        Allowed inputs:
          - Photons: 'gam'
          - Leptons: 'el' (e-), 'posi' (e+)
          - Neutrinos: 'nu_e', 'anu_e', 'nu_mu', 'anu_mu', 'nu_all'
          - Nucleons: 'p', 'ap' (antiproton), 'n', 'an' (antineutron)
          - Antinuclei: 'ad' (antideuteron), 'ah' (antihelium-3)
    primary_target : str
        Primary beam and target combination (e.g., 'p-p', 'p-He', 'He-p',
        'He-He', 'C-p', 'Al-p', 'Fe-p', 'ap-p', 'ap-He').

    Returns
    -------
    Ep_grid : numpy.ndarray (1D)
        Primary energy grid in eV.
    Es_grid : numpy.ndarray (1D)
        Secondary energy grid in eV.
    cs_slice : numpy.ndarray (2D)
        Tabulated invariant cross-section matrix (N_Ep x N_Es).
    """
    species_map = {
        'gam': 0, 'el': 0, 'nu_e': 0, 'p': 0, 'n': 0, 'ad': 0, 'ah': 0,
        'posi': 1, 'anu_e': 1, 'ap': 1, 'an': 1,
        'nu_mu': 2,
        'anu_mu': 3,
        'nu_all': 100
    }

    if secondary not in species_map:
        print(f"Unknown product '{secondary}'. Check your input, please!")
        return None

    data_col = species_map[secondary]

    file_sec_map = {
        'gam': 'gam', 'el': 'el', 'posi': 'el',
        'nu_e': 'nu', 'anu_e': 'nu', 'nu_mu': 'nu', 'anu_mu': 'nu', 'nu_all': 'nu',
        'p': 'pap', 'ap': 'pap',
        'n': 'nan', 'an': 'nan',
        'ad': 'ad', 'ah': 'ah'
    }
    file_sec = file_sec_map[secondary]

    try:
        primary, target = primary_target.split('-')
    except ValueError:
        print(f"Invalid primary_target format '{primary_target}'. Expected format 'p-p', 'He-p', etc.")
        return None

    base_name = f"{file_sec}_{primary}_{target}"
    archive_path = os.path.join(AAFrag_path, f"aafrag_{primary}_data.npz")

    try:
        archive = np.load(archive_path)
    except OSError:
        print(f"There is no data archive for primary particle: '{primary}' ({archive_path})")
        return None

    try:
        Ep_grid = archive[f"{base_name}_Ep"]
        Es_grid = archive[f"{base_name}_Es"]
        cs_grid = archive[f"{base_name}_cs"]
    except KeyError:
        print(f"There is no data for combination '{primary_target}' and secondary '{secondary}'.")
        return None

    if data_col != 100:
        cs_slice = cs_grid[:, :, data_col]
    else:
        cs_slice = cs_grid[:, :, 0:4].sum(axis=2)

    return Ep_grid, Es_grid, cs_slice


###############################################################################
###############################################################################


def get_cs_value(secondary, primary_target, E_primaries, E_secondaries=None):
    """
    Return single differential cross-section vector.

    Parameters
    ----------
    secondary : str
        Secondary particle produced in the interaction.
        Allowed inputs: 'gam', 'posi', 'el', 'nu_e', 'anu_e', 'nu_mu', 'anu_mu',
        'nu_all', 'p', 'ap', 'n', 'an', 'ad', 'ah'.
    primary_target : str
        Primary/target combination (e.g., 'p-p', 'p-He', 'He-p', 'He-He',
        'C-p', 'Al-p', 'Fe-p', 'ap-p', 'ap-He').
    E_primaries : int or float
        Total energy of primary particle in GeV.
    E_secondaries : int, float, list, tuple, or numpy.ndarray, optional
        Secondary particle energy in GeV. Default tabulated binning is used if None.

    Returns
    -------
    numpy.ndarray (2D)
        Array of shape (2, N) containing [dSigma/dE (mb/GeV), E_secondary (GeV)].
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
                      E_secondaries=None, outside_bounds='raise'):
    """
    Reconstruct cross-section values for given values of total primary and
    secondary particle energies.

    Calculates the differential cross-section matrix using physical
    curve-morphing log-log interpolation.

    Parameters
    ----------
    secondary : str
        Secondary particle produced in the interaction.
        Allowed inputs: 'gam', 'posi', 'el', 'nu_e', 'anu_e', 'nu_mu', 'anu_mu',
        'nu_all', 'p', 'ap', 'n', 'an', 'ad', 'ah'.
    primary_target : str
        Primary/target combination ('p-p', 'p-He', 'He-p', 'He-He', 'C-p',
        'Al-p', 'Fe-p', 'ap-p', 'ap-He').
    E_primaries : int, float, list, tuple, or numpy.ndarray, optional
        Vector of primary particle total energy in GeV of size M.
        The default values are taken from the tabulated grid.
    E_secondaries : int, float, list, tuple, or numpy.ndarray, optional
        Vector of secondary particle total energy in GeV of size N.
        The default values are taken from the tabulated grid.
    outside_bounds : str, optional
        Defines behavior if E_primaries is outside tabulated range.
        Allowed inputs:
          - 'raise' : Raises ValueError.
          - 'nans' / 'nan' : Fills out-of-range rows with np.nan.
          - 'zeros' / 'zero' : Fills out-of-range rows with 0.0.
        Default is 'raise'.

    Returns
    -------
    cs_matrix : numpy.ndarray (2D)
        Matrix MxN of differential cross-sections (in mb/GeV).
    energy_primary : numpy.ndarray (1D)
        Vector of primary total energy in GeV.
    energy_secondary : numpy.ndarray (1D)
        Vector of secondary energy in GeV.
    """
    data = open_data_files(secondary, primary_target)
    if data is None:
        return None

    Ep_grid, Es_grid, cs_slice = data

    # 1. Input grid formatting (convert GeV inputs to eV for internal grid alignment)
    if E_primaries is None:
        req_Ep = Ep_grid
        E_primaries = Ep_grid / 1e9
    else:
        E_primaries = np.atleast_1d(E_primaries).flatten()
        req_Ep = E_primaries * 1e9

    if E_secondaries is None:
        if secondary in ['ad', 'ah']:
            # Antinuclei have specialized discrete grids
            default_sec = True
            req_Es = Es_grid
            E_secondaries = Es_grid / 1e9
        else:
            # Universal master grid: 10^-5 to 10^12 GeV (1701 points, 100 pts/decade)
            default_sec = False
            E_secondaries = np.logspace(-5, 12, 1701)
            req_Es = E_secondaries * 1e9
    else:
        default_sec = False
        E_secondaries = np.atleast_1d(E_secondaries).flatten()
        req_Es = E_secondaries * 1e9

    E_th_b, E_th_t = Ep_grid[0], Ep_grid[-1]
    out_of_range = (req_Ep < E_th_b / 1.001) | (req_Ep > E_th_t * 1.001)

    if np.any(out_of_range):
        if outside_bounds == 'raise':
            raise ValueError(
                f"Primary total energy is not in range: {E_trans(E_th_b)} -- "
                f"{E_trans(E_th_t)} available for combination: {primary_target}"
            )
        elif outside_bounds in ['nans', 'nan', 'zeros', 'zero']:
            pass
        else:
            print(
                f"Primary total energy is not in range: {E_trans(E_th_b)} -- "
                f"{E_trans(E_th_t)} available for combination: {primary_target}"
            )
            return None

    log_Ep_grid = np.log10(Ep_grid)
    log_Es_grid = np.log10(Es_grid)
    log_req_Ep = np.log10(req_Ep)
    log_req_Es = np.log10(req_Es)

    with np.errstate(divide='ignore'):
        log_cs_grid = np.log10(cs_slice)

    cs_matrix = np.zeros((len(req_Ep), len(req_Es)))

    # Vectorized search for bounding primary energy brackets
    idx2_all = np.searchsorted(log_Ep_grid, log_req_Ep)
    idx2_all = np.clip(idx2_all, 1, len(log_Ep_grid) - 1)
    idx1_all = idx2_all - 1

    cl1_all = np.abs((log_req_Ep - log_Ep_grid[idx1_all]) / (log_Ep_grid[idx2_all] - log_Ep_grid[idx1_all]))
    cl2_all = np.abs((log_req_Ep - log_Ep_grid[idx2_all]) / (log_Ep_grid[idx2_all] - log_Ep_grid[idx1_all]))

    for i, log_req_E in enumerate(log_req_Ep):
        if out_of_range[i]:
            if outside_bounds in ['nans', 'nan']:
                cs_matrix[i, :] = np.nan
            elif outside_bounds in ['zeros', 'zero']:
                cs_matrix[i, :] = 0.0
            continue

        idx1 = idx1_all[i]
        idx2 = idx2_all[i]
        cl1 = cl1_all[i]
        cl2 = cl2_all[i]

        # Exact grid node match bypass
        if default_sec:
            if np.abs(log_req_E - log_Ep_grid[idx1]) <= np.log10(1.01):
                temp_cs = cs_slice[idx1, :].copy()
                temp_cs[0] = 0.0
                cs_matrix[i, :] = temp_cs
                continue
            elif np.abs(log_req_E - log_Ep_grid[idx2]) <= np.log10(1.01):
                temp_cs = cs_slice[idx2, :].copy()
                temp_cs[0] = 0.0
                cs_matrix[i, :] = temp_cs
                continue

        # Physical curve morphing
        si1_cs = log_cs_grid[idx1, :].copy()
        si2_cs = log_cs_grid[idx2, :].copy()

        # Low energy threshold cleanup
        valid_es = log_Es_grid < 8
        w1 = np.where((si1_cs == -np.inf) & valid_es)[0]
        if len(w1) > 0:
            si1_cs[:w1[-1]] = -np.inf

        w2 = np.where((si2_cs == -np.inf) & valid_es)[0]
        if len(w2) > 0:
            si2_cs[:w2[-1]] = -np.inf

        v1 = si1_cs != -np.inf
        v2 = si2_cs != -np.inf

        a1_x, a1_y = log_Es_grid[v1][1:], si1_cs[v1][1:]
        a2_x, a2_y = log_Es_grid[v2][1:], si2_cs[v2][1:]

        if len(a1_x) == 0 or len(a2_x) == 0:
            cs_matrix[i, :] = 0.0
            if default_sec:
                cs_matrix[i, 0] = 0.0
            continue

        min_a1_x, max_a1_x = a1_x[0], a1_x[-1]
        min_a2_x, max_a2_x = a2_x[0], a2_x[-1]

        new_a1_x = np.linspace(min_a1_x, max_a1_x, 1000)
        new_a2_x = np.linspace(min_a2_x, max_a2_x, 1000)

        new_a1_y = np.interp(new_a1_x, a1_x, a1_y)
        new_a2_y = np.interp(new_a2_x, a2_x, a2_y)

        midx = cl2 * new_a1_x + cl1 * new_a2_x
        midy = cl2 * new_a1_y + cl1 * new_a2_y

        filter_energies = (
            (log_req_Es > (min_a1_x if min_a1_x < min_a2_x else min_a2_x) - 1e-5) &
            (log_req_Es < (max_a1_x if max_a1_x > max_a2_x else max_a2_x) + 1e-5) &
            (log_req_Es <= log_req_E + 1e-5) &
            (log_req_Es <= midx[-1] + 1e-5) &
            (log_req_Es >= midx[0] - 1e-5)
        )

        sigma_final = np.full(len(req_Es), -np.inf)
        if np.any(filter_energies):
            sigma_final[filter_energies] = np.interp(log_req_Es[filter_energies], midx, midy)

        cs_matrix[i, :] = np.power(10, sigma_final)

        if default_sec:
            cs_matrix[i, 0] = 0.0

    # Convert invariant cross section to differential dSigma/dE (mb/GeV)
    cs_matrix = cs_matrix / E_secondaries[None, :]

    if len(E_primaries) == 1:
        return np.array([cs_matrix]), np.array([E_primaries]), E_secondaries

    return cs_matrix, np.squeeze(E_primaries), E_secondaries


###############################################################################
###############################################################################


def get_spectrum(energy_primary, energy_secondary, cs_matrix, prim_spectrum):
    """
    Calculate the spectrum of secondary particles.

    Generates differential secondary particle spectrum for given
    secondary differential cross-section matrix and primary spectrum.

    Parameters
    ----------
    energy_primary : numpy.ndarray (1D)
        Vector of primary energies, GeV.
    energy_secondary : numpy.ndarray (1D)
        Vector of secondary energies, GeV.
    cs_matrix : numpy.ndarray (2D)
        Matrix MxN of differential cross-sections in mb/GeV.
    prim_spectrum : numpy.ndarray (1D or 2D)
        Primary particle differential spectrum.

    Returns
    -------
    numpy.ndarray (1D)
        Differential spectrum of secondary particles.
    """
    if len(prim_spectrum.shape) == 2:
        prim_spectrum = prim_spectrum[:, 0]

    E1 = energy_primary[:-1, np.newaxis]
    E2 = energy_primary[1:, np.newaxis]
    Y1 = cs_matrix[:-1] * prim_spectrum[:-1, np.newaxis]
    Y2 = cs_matrix[1:] * prim_spectrum[1:, np.newaxis]

    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(Y2 / Y1) / np.log(E2 / E1) + 1.0
        INT = (Y2 * E2 - Y1 * E1) / log_term
        INT[np.isnan(INT)] = 0.0

    return np.sum(INT, axis=0)


###############################################################################
###############################################################################


def get_cross_section_Kafexhiu2014(E_primaries, E_secondaries):
    """
    Return cross-section values (Kafexhiu et al. 2014).

    Return the matrix of the differential cross-section for a given
    combination of energy vectors, primary energy vector, secondary energy
    vector.

    Based on Kafexhiu et al. 2014 (GEANT parameters).
    Calculations are performed for p-p interactions and gamma production only.
    Works well at low energies, but should be substituted by newer codes at high energies.

    Parameters
    ----------
    E_primaries : int, float, list, tuple, or numpy.ndarray
        Vector of the primary proton energy (in GeV) of size M.
    E_secondaries : int, float, list, tuple, or numpy.ndarray
        Vector of the gamma energy (in GeV) of size N.

    Returns
    -------
    numpy.ndarray (2D)
        Matrix MxN of the differential cross-section (in mb/GeV)
        for a given combination of vectors.
    numpy.ndarray (1D)
        Vector of primary energy in GeV.
    numpy.ndarray (1D)
        Vector of secondary energy in GeV.
    """
    from Kafexhiu2014 import F_gamma_Kafexhiu2014
    csf = np.vectorize(F_gamma_Kafexhiu2014)

    if (E_primaries is None) or (E_secondaries is None):
        print('Error: please provide the energy binning for protons and secondary particles.')
        return None

    E_primaries = np.atleast_1d(E_primaries)
    E_secondaries = np.atleast_1d(E_secondaries)
    cs_matrix = np.zeros([len(E_primaries), len(E_secondaries)])

    for i, E_p in enumerate(E_primaries):
        cs_matrix[i] = csf(E_p - m_p, E_secondaries, 'GEANT')

    return cs_matrix, E_primaries, E_secondaries


###############################################################################
###############################################################################


def get_cross_section_Kamae2006(secondary, E_primaries,
                                E_secondaries, diffractive=True):
    """
    Return cross-section values (Kamae et al. 2006).

    Return the matrix of the differential cross-section for a given
    combination of energy vectors, primary energy vector, secondary energy
    vector.

    Based on Kamae et al. 2006.
    Calculations are performed for p-p interactions and for gamma and lepton
    production only. Works well at low energies, but should be substituted by
    newer codes at high energies.

    Parameters
    ----------
    secondary : str
        Secondary particle of proton-proton interaction ('gam', 'el', 'posi',
        'nu_e', 'anu_e', 'nu_mu', 'anu_mu', 'nu_all').
    E_primaries : int, float, list, tuple, or numpy.ndarray
        Vector of the primary proton energy (in GeV) of size M.
    E_secondaries : int, float, list, tuple, or numpy.ndarray
        Vector of the secondary particle energy (in GeV) of size N.
    diffractive : bool, optional
        Include or exclude diffractive processes. Default is True.

    Returns
    -------
    numpy.ndarray (2D)
        Matrix MxN of the differential cross-section (in mb/GeV)
        for a given combination of vectors.
    numpy.ndarray (1D)
        Vector of primary energy in GeV.
    numpy.ndarray (1D)
        Vector of secondary energy in GeV.
    """
    if secondary == 'gam':
        from Kamae2006 import dXSdE_gamma_Kamae2006 as model_func
    elif secondary == 'el':
        from Kamae2006 import dXSdE_elec_Kamae2006 as model_func
    elif secondary == 'posi':
        from Kamae2006 import dXSdE_posi_Kamae2006 as model_func
    elif secondary == 'nu_e':
        from Kamae2006 import dXSdE_elec_nu_Kamae2006 as model_func
    elif secondary == 'anu_e':
        from Kamae2006 import dXSdE_elec_anti_nu_Kamae2006 as model_func
    elif secondary == 'nu_mu':
        from Kamae2006 import dXSdE_mu_nu_Kamae2006 as model_func
    elif secondary == 'anu_mu':
        from Kamae2006 import dXSdE_mu_anti_nu_Kamae2006 as model_func
    elif secondary == 'nu_all':
        from Kamae2006 import (
            dXSdE_elec_nu_Kamae2006, dXSdE_elec_anti_nu_Kamae2006,
            dXSdE_mu_nu_Kamae2006, dXSdE_mu_anti_nu_Kamae2006
        )

        def model_func(T_p, T_secondaries, diffractive):
            return (
                dXSdE_elec_nu_Kamae2006(T_p, T_secondaries, diffractive) +
                dXSdE_elec_anti_nu_Kamae2006(T_p, T_secondaries, diffractive) +
                dXSdE_mu_nu_Kamae2006(T_p, T_secondaries, diffractive) +
                dXSdE_mu_anti_nu_Kamae2006(T_p, T_secondaries, diffractive)
            )
    else:
        def model_func(T_secondaries, T_primaries):
            return np.zeros(len(T_secondaries))

    csf = np.vectorize(model_func)

    if (E_primaries is None) or (E_secondaries is None):
        print('Error: please provide the energy binning for protons and secondary particles.')
        return None

    E_primaries = np.atleast_1d(E_primaries)
    E_secondaries = np.atleast_1d(E_secondaries)
    cs_matrix = np.zeros([len(E_primaries), len(E_secondaries)])

    for i, E_p in enumerate(E_primaries):
        if E_p < 512e3:
            cs_matrix[i] = csf(E_p - m_p, E_secondaries, diffractive)

    return cs_matrix, E_primaries, E_secondaries