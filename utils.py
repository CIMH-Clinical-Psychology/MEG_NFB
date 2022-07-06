# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:25:30 2022

@author: Simon Kern
"""
import mne
import numpy as np
from mne.io.pick import _picks_to_idx, pick_info

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