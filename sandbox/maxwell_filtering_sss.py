"""

Signal Space Separation test run with head movement compensation
Test data:
    Rest_raw.fif   (resting state measurement)
    sss_cal.dat    (MEG sensor fine calibration)
    ct_sparse.fif  (crosstalk information)

ZI Mannheim

"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import find_bad_channels_maxwell
from cpu_usage import CPUUsageLogger

if __name__=='__main__':
    cpu = CPUUsageLogger()
    
    #%% Read data from files
    sample_data_raw_file = 'Z:/imagine.fif'
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose = False)
    # raw.crop(tmax = 362)
    
    fine_cal_file = './calibration_files/sss_cal.dat'
    crosstalk_file = './calibration_files/ct_sparse.fif'
    
    #%% Find bad channels
    cpu.start('find bad channels')
    
    raw.info['bads'] = []
    raw_check = raw.copy()
    
    print("========START BAD CHANNEL DETETCTION========\n")
    
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(raw_check, cross_talk = crosstalk_file, calibration = fine_cal_file, return_scores = True, verbose = True)
    print("Noisy channels:", auto_noisy_chs)
    print("Flat channels:", auto_flat_chs)
    
    print("========END BAD CHANNEL DETETCTION========\n")
    
    # Now update the list of bad channels in the dataset
    
    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads
    
    
    #%% Calculate head position
    
    print("========START HPI PROCESSING========\n")
    cpu.set_segment_name('estimate cHPI amplitudes')
    
    # Estimate HPI coil amplitudes
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose = True)
    
    # Calculate HPI coil locations from amplitudes
    cpu.set_segment_name('compute cHPI locs & headpos')
    
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose = True)
    
    # Calculate continuous head position from HPI coil amplitudes and locations
    head_position = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose = True)
    
    # Plot continuous head position
    # mne.viz.plot_head_positions(head_position, mode = 'traces')
    
    print("========END HPI PROCESSING========\n")
    
    #%% Apply Maxwell filtering
    
    print("========START MAXWELL FILTERING========\n")
    cpu.set_segment_name('maxfilter')
    
    raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto', verbose = True, int_order = 8, ext_order = 4)
    
    print("========END MAXWELL FILTERING========\n")
    cpu.plot(block=True)
    #%% Various plots
    
    # print("========START PLOTTING========\n")
    
    # # Compare unfiltered and filtered data
    # raw.pick(['meg']).plot(scalings = dict(mag = 1e-10, grad = 6e-9), clipping = 10, start = 100, duration = 0.5, butterfly = True, title = 'Before SSS')
    # raw_sss.pick(['meg']).plot(scalings = dict(mag = 1e-10, grad = 3e-9), clipping = 10, start = 100, duration = 0.5, butterfly = True, title = 'After SSS')
    
    # # raw.pick_types(meg = 'mag').plot(scalings = 'auto', start = 100, duration = 0.1, butterfly = True)
    # # raw.pick_types(meg = 'grad').plot(scalings = 'auto', start = 100, duration = 0.1, butterfly = True) # Warum gibt es keine channels vom Typ grad?
    
    # # Compare unfiltered/filtered power spectral density
    # raw.plot_psd()
    # raw_sss.plot_psd()
    
    # raw.plot(scalings = dict(mag = 8e-10, grad = 1e-9), clipping = 2, start = 100, duration = 0.5)
    # # raw.plot_sensors()
    
    # print("========END PLOTTING========\n")