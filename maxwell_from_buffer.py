"""
        Accessing FieldTrip buffer with python while running FieldTrip fileproxy with pre-recorded data
        FieldTrip fileproxy is running in MATLAB
        Buffer = localhost:1972
"""

import sys

import mne
from mne.preprocessing import maxwell_filter, find_bad_channels_maxwell
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos

from fieldtrip import FieldTrip as ft

from scipy import signal

import matplotlib.pyplot as plt

#%% Load calibration and crosstalk files

fine_cal_file = 'C:/Users/Mario/Documents/_Studium/BA/NFB/data/sss_cal.dat'
crosstalk_file = 'C:/Users/Mario/Documents/_Studium/BA/NFB/data/ct_sparse.fif'

sample_data_raw_file = 'C:/Users/Mario/Documents/_Studium/BA/NFB/data/Rest_raw.fif'
raw_from_file = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose = False)

#%% Connect to buffer

ftc = ft.Client()
ftc.connect('localhost', 1972)

#%% Retrieve header of latest chunk in buffer

header = ftc.getHeader()

# Check for errors
if header is None:
    print('Failed to retrieve header!')
    sys.exit(1)

# Print general header information (number of channels, sample frequency etc.)
print(header)

#%% Create Info object

sampling_freq = 1000.0
# bads = find_bad_channels()        # To be implemented

# info_obj = mne.create_info(header.labels, sfreq = sampling_freq, verbose = True)

info_obj = raw_from_file.info

# info_obj['bads'] = ['MEG1042', 'MEG1421']                           # Test
# info_obj['lowpass'] = raw_from_file.info['lowpass']
# info_obj['highpass'] = raw_from_file.info['highpass']
# info_obj['description'] = raw_from_file.info['description']
# info_obj['hpi_subsystem'] = raw_from_file.info['hpi_subsystem']
# info_obj['hpi_meas'] = raw_from_file.info['hpi_meas']

print(info_obj)

#%% Retrieve latest chunks from buffer and create Raw objects

chunk = []
raws_sss_list = []

header = ftc.getHeader()
newSamples = header.nSamples
oldSamples = header.nSamples

maxSamples = 362999

while header.nSamples <= maxSamples:        # Maximum number of samples = 363000
    if newSamples >= 1000:
        
        print('#######################################')
        print('##########  NEW CHUNK  ################')
        print('#######################################\n')
        print('Trying to read last chunk (1000 samples)...')
        
        startIndex = header.nSamples - 1000
        stopIndex = header.nSamples - 1
        print('Start sample: ', startIndex, '\n', 'Stop sample: ', stopIndex, '\n')
        
        chunk = ftc.getData([startIndex, stopIndex])
        # print('Chunk = ', chunk, '\n' * 3)
        
        raw = mne.io.RawArray(chunk.T, info_obj, verbose = False)
        
        ##### Bad channel detection
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
        
        print('End cHPI processing.\n')
        
        ##### Run Maxwell filtering on current chunk
        print('Start Maxwell filtering...\n')
        raw_sss = maxwell_filter(raw, int_order = 8, ext_order = 3, head_pos = head_position, calibration = fine_cal_file, cross_talk = crosstalk_file, mag_scale ='auto', verbose = False)
        print('End Maxwell filtering.\n')
        
        # Append list of Maxwell-filtered Raw objects
        raws_sss_list.append(raw_sss)
    
    # Calculate new sample interval
    header = ftc.getHeader()
    newSamples = header.nSamples - oldSamples
    oldSamples = header.nSamples

# Concatenate Maxwell-filtered list of Raw objects into single instance
print('Concatenate list of Maxwell-filtered Raw objects to single instance of type Raw...\n')
raw_final = mne.concatenate_raws(raws_sss_list, preload = True, events_list = None, verbose = True)
#raw_final = mne.Annotations.delete(idx)                    # Delete boundary annotations?

#%% Offline SSS for comparison

# Estimate HPI coil amplitudes
chpi_amplitudes_offline = mne.chpi.compute_chpi_amplitudes(raw_from_file, verbose = False)

# Calculate HPI coil locations from amplitudes
chpi_locs_offline = mne.chpi.compute_chpi_locs(raw_from_file.info, chpi_amplitudes_offline, verbose = False)

# Calculate continuous head position from HPI coil amplitudes and locations
head_position_offline = mne.chpi.compute_head_pos(raw_from_file.info, chpi_locs_offline, verbose = False)

raw_sss_offline = maxwell_filter(raw_from_file, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position_offline, mag_scale = 'auto', verbose = False, int_order = 8, ext_order = 3)

#%% Compute and plot time series correlation

signal_matrix_online = raw_final.get_data()
signal_matrix_offline = raw_sss_offline.get_data()

# Calculate cross-correlation between online and offline analysis
signal_correlation = signal.correlate(signal_matrix_online, signal_matrix_offline, mode = 'full', method = 'fft')

# Define plot parameters
fig, (ax_online, ax_offline, ax_corr) = plt.subplots(3, 1, sharex = True)

ax_online.plot(signal_matrix_online)
ax_offline.set_title('Online signal')

ax_offline.plot(signal_matrix_offline)
ax_offline.set_title('Offline signal')

ax_corr.plot(signal_correlation)
ax_corr.set_title('Cross-correlation')

fig.tight_layout()
plt.show()

#%% Disconnect client from buffer

#ftc.disconnect()