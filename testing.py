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
def headpos(raw, t_step_max=1.0):
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose = False)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, t_step_max=t_step_max, verbose = False)
    head_position = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose = False)
    return head_position

fine_cal_file = './calibration_files/sss_cal.dat'
crosstalk_file = './calibration_files/ct_sparse.fif'
sample_data_raw_file = 'Z:/rest.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose = False)
raw = raw.crop(tmax = 15)
glob = maxfilter_realtime.glob



#%% Calculate head position
# calculate with original headpos functions for comparing results

t_step_max = 0.5

MaxFilterClient = maxfilter_realtime.MaxFilterClient(raw.copy())

stimer.start()
headpos_live = MaxFilterClient.headpos(t_step_max=t_step_max)   
t2 = stimer.stop()

stimer.start()
headpos_orig = headpos(raw.copy(), t_step_max=t_step_max)
t1 = stimer.stop()

np.testing.assert_array_equal(headpos_live, headpos_orig)

print(f'{t1:.2f}s - {t2:.2f}s = {t1-t2:.2f}s faster')
stop

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

















