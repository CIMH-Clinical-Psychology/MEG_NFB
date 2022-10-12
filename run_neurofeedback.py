# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:02:10 2022

This script starts a new Neurofeedback client that connects
to a streaming server (e.g. FieldTripBuffer or LSLServer, )

@author: Simon Kern
"""
import os
import mne
import time
import matplotlib
try:
    # somebody (probably me) put this here, but I don't remember why.
    matplotlib.use('TkAgg')
except:
    pass

from mne_realtime import LSLClient, MockLSLStream, FieldTripClient
import numpy as np
import matplotlib.pyplot as plt
import utils
from joblib.memory import Memory
from maxfilter_realtime import MaxFilterClient, MaxFilterCoordinator
from externals.FieldTrip import FieldTripClient
from externals.FieldTrip import FieldTripClientSimulator
from mne.preprocessing import maxwell_filter


# use caching for reading the raw file
mem = Memory(os.path.expanduser('~/mne-nfb/cache/')) 
mne.io.read_raw = mem.cache(mne.io.read_raw)
fine_cal_file = './calibration_files/sss_cal.dat'
crosstalk_file = './calibration_files/ct_sparse.fif'

#%%

from mne.io.pick import (pick_types, pick_channels, pick_channels_regexp,
                      pick_info)

from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
from mne.preprocessing.maxwell import maxwell_filter, find_bad_channels_maxwell

from mne.chpi import  _fit_chpi_amplitudes, _time_prefix, _fit_chpi_quat_subset
from mne.chpi import _get_hpi_initial_fit, _fit_magnetic_dipole, _check_chpi_param
from mne.io.pick import pick_types, pick_channels, pick_channels_regexp, pick_info
from mne.forward import _create_meg_coils,  _concatenate_coils
from mne.dipole import _make_guesses
from mne.preprocessing.maxwell import _sss_basis, _prep_mf_coils
from mne.preprocessing.maxwell import _get_mf_picks_fix_mags, _regularize_out
from mne.transforms import apply_trans, invert_transform, _angle_between_quats
from mne.transforms import quat_to_rot, rot_to_quat, _fit_matched_points
from mne.transforms import _quat_to_affine
from mne.utils import (verbose, logger, use_log_level, _check_fname, warn,
                    _validate_type, ProgressBar, _check_option)
from multiprocessing import Process, Queue, shared_memory
from multiprocessing.shared_memory import SharedMemory
from threading import Thread

#%% MAIN
if __name__=='__main__':
    
    # SETTINGS

    fif_file = utils.get_demo_rest_fif() # a fif file that will be streamed by the simulator

    LSLClient.get_data_as_raw = utils.get_data_as_raw

    raw = mne.io.read_raw(fif_file, preload=True, verbose='WARNING')
    raw.del_proj('all') # else they get copied to the layer, even though they are not applied
    
    # maxfilter signal just for comparison
    with stimer:

        auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, verbose = False)
        bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
        raw.info['bads'] = bads
        
        ##### cHPI processing
        print('Start cHPI processing...\n')
        
        chpi_amplitudes = compute_chpi_amplitudes(raw, verbose = False)
    
        # Calculate HPI coil locations from amplitudes
        chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes, verbose = False)
    
        # Calculate continuous head position from HPI coil amplitudes and locations
        head_position = compute_head_pos(raw.info, chpi_locs, verbose = False)
        raw_sss = maxwell_filter(raw, int_order = 8, ext_order = 3,
                                   head_pos = head_position, 
                                 calibration = fine_cal_file,
                                 cross_talk = crosstalk_file, 
                                 mag_scale ='auto', 
                                 verbose = False)
    
    # raw.resample(100)
    
    # stream = MockLSLStream('mock', raw_orig, 'all', time_dilation=2).start()
    # client = LSLClient(info=raw_orig.info, host='mock', wait_max=10).start()
    # client = FieldTripClient('localhost')
    # start a FieldTripBuffer Simulator. It will feed 100 samples/second.
    data_client = FieldTripClientSimulator(raw=raw, sfreq=1000)
    
    # start the data client that is the central hub of retrieving
    # and storing the data from the MEG into our shared memory
    maxfilter_client = MaxFilterClient(raw)
    
    
    bufferlen = 15
    coordinator = MaxFilterCoordinator(data_client, maxfilter_client,
                                       chunklen=1, bufferlen=bufferlen)
    
    coordinator.start()
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.plot(raw.get_data([50], tmax=bufferlen)[0], color='b', linewidth=0.1)
    ax1.plot(raw_sss.get_data([50], tmax=bufferlen)[0], color='r', linewidth=0.1)
    
    for i in range(100):
        print('script __main__ running...')
        # print(type(dataclient.times))
        with coordinator.lock:
            data = coordinator.data
            data_filtered = maxfilter_client.maxfilter_monolithic(data)
            
        ax2.clear()
        ax2.plot(data[50], color='b', linewidth=0.1)
        ax2.plot(data_filtered[50], color='r', linewidth=0.1)

        
        # plt.show(block=False)
        plt.pause(1)
    # self = LiveProcessor(client, movement_corr=True)
    
