#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:34:13 2026

@author: sergeykoldobskiy
"""

import pytest
import numpy as np
from aafragpy import aafrag
from aafragpy import Kamae2006, Kafexhiu2014
import os

STANDARD_SPECIES = [
    'gam', 'el', 'posi', 
    'nu_e', 'anu_e', 'nu_mu', 'anu_mu', 'nu_all', 
    'p', 'ap', 'n', 'an'
]
ANTINUCLEI = ['ad', 'ah']

VALID_CHANNELS = []

# p and He projectiles (56 channels) @ 100 GeV
for prim in ['p', 'He']:
    for tgt in ['p', 'He']:
        ch = f"{prim}-{tgt}"
        for sec in STANDARD_SPECIES + ANTINUCLEI:
            VALID_CHANNELS.append((sec, ch, 100.0))

# Heavy ions on proton targets (36 channels) @ > 15 GeV/nucleon
# C-p @ 500 GeV (41.7 GeV/n), Al-p @ 1000 GeV (37.0 GeV/n), Fe-p @ 2000 GeV (35.7 GeV/n)
for prim, ep_test in [('C', 500.0), ('Al', 1000.0), ('Fe', 2000.0)]:
    for sec in STANDARD_SPECIES:
        VALID_CHANNELS.append((sec, f"{prim}-p", ep_test))

# Antiproton beams on p and He targets (4 channels) @ 100 GeV
for tgt in ['p', 'He']:
    for sec in ANTINUCLEI:
        VALID_CHANNELS.append((sec, f"ap-{tgt}", 100.0))

assert len(VALID_CHANNELS) == 96, f"Expected 96 channels, found {len(VALID_CHANNELS)}"


@pytest.mark.parametrize("sec, channel, ep_test", VALID_CHANNELS)
def test_channel_evaluation(sec, channel, ep_test):
    """Verify that every valid channel returns a non-empty, finite, positive cross-section."""
    res = aafrag.get_cross_section(sec, channel, E_primaries=ep_test)
    assert res is not None, f"Failed to load channel {sec} for {channel}"
    
    cs_matrix, ep_vec, es_vec = res
    cs_vals = cs_matrix.squeeze()
    
    assert len(es_vec) > 0
    assert not np.isnan(cs_vals).any()
    assert np.nanmax(cs_vals) > 0.0


def test_invalid_channels():
    """Verify that invalid combinations safely return None."""
    assert aafrag.get_cross_section('gam', 'C-He', E_primaries=100.0) is None
    assert aafrag.get_cross_section('gam', 'ap-p', E_primaries=100.0) is None


def test_out_of_bounds_exception():
    """Verify that sub-threshold energies trigger ValueError."""
    with pytest.raises(ValueError):
        aafrag.get_cross_section('gam', 'p-p', E_primaries=1.0, outside_bounds='raise')
        
def test_vectorized_energy_inputs():
    """Verify handling of multi-energy primary arrays."""
    ep_grid = np.logspace(1, 4, 15)  # 15 energy points
    res = aafrag.get_cross_section('gam', 'p-p', E_primaries=ep_grid)
    cs_matrix, ep_out, es_out = res
    assert cs_matrix.shape == (15, len(es_out))
    assert not np.isnan(cs_matrix).any()

def test_spectrum_integration():
    """Verify get_spectrum integration produces positive secondary flux."""
    ep_grid = np.logspace(1, 4, 30)
    es_grid = np.logspace(-1, 3, 50)
    cs_matrix, _, _ = aafrag.get_cross_section('gam', 'p-p', ep_grid, es_grid)
    flux = aafrag.get_spectrum(ep_grid, es_grid, cs_matrix, ep_grid**(-2.7))
    assert len(flux) == len(es_grid)
    assert np.all(flux >= 0.0)
    assert np.max(flux) > 0.0

def test_external_models():
    """Verify Kamae2006 and Kafexhiu2014 modules load and evaluate."""
    
    
    # Kafexhiu gamma
    res_kaf = aafrag.get_cross_section_Kafexhiu2014(E_primaries=10.0, E_secondaries=np.logspace(-1, 1, 20))
    assert res_kaf is not None
    assert np.max(res_kaf[0]) > 0.0

    # Kamae gamma & positron
    res_kam_g = aafrag.get_cross_section_Kamae2006('gam', E_primaries=50.0, E_secondaries=np.logspace(-1, 1, 20))
    res_kam_p = aafrag.get_cross_section_Kamae2006('posi', E_primaries=50.0, E_secondaries=np.logspace(-1, 1, 20))
    assert np.max(res_kam_g[0]) > 0.0
    assert np.max(res_kam_p[0]) > 0.0
    


REF_FILE = os.path.join(os.path.dirname(__file__), "ref_benchmarks.npz")

@pytest.mark.skipif(not os.path.exists(REF_FILE), reason="Reference benchmark file not generated.")
def test_numerical_regression_digits():
    """Compare cross-section digits against stored reference values."""
    ref_data = np.load(REF_FILE)
    es_grid = np.logspace(-1, 2, 50)
    
    # Check gammas @ 100 GeV (match within 0.01% relative tolerance)
    cs_gam, _, _ = aafrag.get_cross_section('gam', 'p-p', 100.0, es_grid)
    np.testing.assert_allclose(
        cs_gam, ref_data['gam_pp_100GeV'], 
        rtol=1e-4, atol=1e-15,
        err_msg="Gamma p-p cross-section values diverged from reference baseline!"
    )
    
    # Check antiprotons @ 1 TeV
    cs_ap, _, _ = aafrag.get_cross_section('ap', 'p-p', 1000.0, es_grid)
    np.testing.assert_allclose(
        cs_ap, ref_data['ap_pp_1000GeV'], 
        rtol=1e-4, atol=1e-15,
        err_msg="Antiproton p-p cross-section values diverged from reference baseline!"
    )