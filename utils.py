# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:25:30 2022

@author: Simon Kern
"""
import os

import pooch
import mne
import numpy as np
from mne.preprocessing import maxwell_filter
from mne.io.pick import _picks_to_idx, pick_info
from joblib.memory import Memory

# use caching for reading the raw file
mem = Memory(os.path.expanduser('~/mne-nfb/cache/')) 
mne.io.read_raw = mem.cache(mne.io.read_raw)


def get_demo_rest_fif():
    """
    Download the demo file
    which is a 6 minute resting state 
    session, using pooch

    Returns
    -------
    file : str
        path of the downloaded fif file.

    """
    url = 'https://cloud.skjerns.de/index.php/s/Boz57e2ACGNnxXR/download/rest.fif'
    path = os.path.expanduser('~/.meg_nfb/')
    
    file = pooch.retrieve(url, 'md5:d22d4fb58e654f4f5255aeeaaec79d82',
                          fname='rest.fif',
                          path=path, progressbar=True)
    return file


def create_comparison_files():
    """
    creates reference SSS-filtered files for the demo file

    Returns
    -------
    None.

    """
    crosstalk_file = './calibration_files/ct_sparse.fif'
    fine_cal_file = './calibration_files/sss_cal.dat'    
    
    fif_file = get_demo_rest_fif()
    
    save_to = f'{fif_file[:-4]}_sss_mc.fif'
    if os.path.exists(save_to):
        return save_to
    
    raw = mne.io.read_raw(fif_file, preload=True)
    
    # Estimate HPI coil amplitudes
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)

    # Calculate HPI coil locations from amplitudes
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)

    # Calculate continuous head position from HPI coil amplitudes and locations
    head_position = mne.chpi.compute_head_pos(raw.info, chpi_locs)

    raw_sss = maxwell_filter(raw, cross_talk = crosstalk_file, 
                                     calibration = fine_cal_file, 
                                     head_pos = head_position, 
                                     mag_scale = 'auto', 
                                     int_order = 8, 
                                     ext_order = 3)
    raw_sss.save(save_to)
    return save_to


def epoch2raw(epoch, info=None, verbose='WARNING'):
    """
    Convert an mne.Epochs object with a single epoch into a mne.Raw

    Parameters
    ----------
    epoch : TYPE
        DESCRIPTION.

    Returns
    -------
    raw : TYPE
        DESCRIPTION.

    """
    assert len(epoch)==1, 'epoch object contains more than 1 epoch'
    data = epoch.get_data().squeeze()
    if info is None:
        info = epoch.info
    raw = mne.io.RawArray(data, info, verbose = verbose)

    return raw



def get_data_as_raw(self, n_samples=1024, picks=None, verbose='WARNING'):
    """Return last n_samples from current time.

    Parameters
    ----------
    n_samples : int
        Number of samples to fetch.
    %(picks_all)s

    Returns
    -------
    epoch : instance of Epochs
        The samples fetched as an Epochs object.

    See Also
    --------
    mne.Epochs.iter_evoked
    """
    # set up timeout in case LSL process hang. wait arb 5x expected time
    wait_time = n_samples * 5. / self.info['sfreq']

    samples, _ = self.client.pull_chunk(max_samples=n_samples,
                                        timeout=wait_time)
    data = np.vstack(samples).T

    picks = _picks_to_idx(self.info, picks, 'all', exclude=())
    info = pick_info(self.info, picks)
    return mne.io.RawArray(data[picks], info, verbose=verbose)