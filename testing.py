# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:29:30 2021

@author: Simon Kern
"""

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
import tqdm
import stimer
from deepdiff import DeepDiff
from mne.preprocessing import find_bad_channels_maxwell
import maxfilter_realtime
#%% Read data from files
fine_cal_file = './calibration_files/sss_cal.dat'
crosstalk_file = './calibration_files/ct_sparse.fif'
sample_data_raw_file = 'Z:/rest.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose = False)
raw = raw.crop(tmax = 2)
glob = maxfilter_realtime.glob

#%% Calculate head position

# Estimate HPI coil amplitudes
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose = True)

# Calculate HPI coil locations from amplitudes
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose = True)

# Calculate continuous head position from HPI coil amplitudes and locations
head_position = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose = True)  


#%% Apply Maxwell filtering


# raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto')
stimer.start()
raw_sss_live1 = maxfilter_realtime.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto')
t1 = stimer.stop()
stimer.start()
raw_sss_live2 = maxfilter_realtime.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto')
t2 = stimer.stop()

raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto')
info2 = glob['info'].copy()

np.testing.assert_array_equal(raw_sss._data, raw_sss_live1._data)
np.testing.assert_array_equal(raw_sss_live1._data, raw_sss_live2._data)

print(f'{t1:.2f}s - {t2:.2f}s = {t1-t2:.2f}s faster')

















