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

#%% Read data from files
sample_data_raw_file = 'Z:/imagine.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, preload = True, verbose = False)
# raw.crop(tmax = 362)

fine_cal_file = './calibration_files/sss_cal.dat'
crosstalk_file = './calibration_files/ct_sparse.fif'

#%% Find bad channels

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

#%% Plot bad channel heatmap

# Only select the data for gradiometer channels.
ch_type = 'grad'
ch_subset = auto_scores['ch_types'] == ch_type
ch_names = auto_scores['ch_names'][ch_subset]
scores = auto_scores['scores_noisy'][ch_subset]
limits = auto_scores['limits_noisy'][ch_subset]
bins = auto_scores['bins']  # The the windows that were evaluated.
# We will label each segment by its start and stop time, with up to 3
# digits before and 3 digits after the decimal place (1 ms precision).
bin_labels = [f'{start:3.3f} – {stop:3.3f}'
              for start, stop in bins]

# We store the data in a Pandas DataFrame. The seaborn heatmap function
# we will call below will then be able to automatically assign the correct
# labels to all axes.
data_to_plot = pd.DataFrame(data = scores, columns = pd.Index(bin_labels, name = 'Time (s)'), index = pd.Index(ch_names, name = 'Channel'))

# First, plot the "raw" scores.
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
fig.suptitle(f'Automated noisy channel detection: {ch_type}',
             fontsize = 16, fontweight = 'bold')
sns.heatmap(data = data_to_plot, cmap = 'Reds', cbar_kws = dict(label = 'Score'),
            ax = ax[0])
[ax[0].axvline(x, ls = 'dashed', lw = 0.25, dashes = (25, 15), color = 'gray')
    for x in range(1, len(bins))]
ax[0].set_title('All Scores', fontweight = 'bold')

# Now, adjust the color range to highlight segments that exceeded the limit.
sns.heatmap(data = data_to_plot, vmin = np.nanmin(limits),  # bads in input data have NaN limits
            cmap = 'Reds', cbar_kws = dict(label = 'Score'), ax = ax[1])
[ax[1].axvline(x, ls = 'dashed', lw = 0.25, dashes = (25, 15), color = 'gray')
    for x in range(1, len(bins))]
ax[1].set_title('Scores > Limit', fontweight = 'bold')

# The figure title should not overlap with the subplots.
fig.tight_layout(rect = [0, 0.03, 1, 0.95])

#%% Calculate head position

print("========START HPI PROCESSING========\n")

# Estimate HPI coil amplitudes
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose = True)

# Calculate HPI coil locations from amplitudes
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose = True)

# Calculate continuous head position from HPI coil amplitudes and locations
head_position = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose = True)

# Plot continuous head position
mne.viz.plot_head_positions(head_position, mode = 'traces')

print("========END HPI PROCESSING========\n")

#%% Apply Maxwell filtering

print("========START MAXWELL FILTERING========\n")

raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk = crosstalk_file, calibration = fine_cal_file, head_pos = head_position, mag_scale ='auto', verbose = True, int_order = 8, ext_order = 4)

print("========END MAXWELL FILTERING========\n")

#%% Various plots

print("========START PLOTTING========\n")

# Compare unfiltered and filtered data
raw.pick(['meg']).plot(scalings = dict(mag = 1e-10, grad = 6e-9), clipping = 10, start = 100, duration = 0.5, butterfly = True, title = 'Before SSS')
raw_sss.pick(['meg']).plot(scalings = dict(mag = 1e-10, grad = 3e-9), clipping = 10, start = 100, duration = 0.5, butterfly = True, title = 'After SSS')

# raw.pick_types(meg = 'mag').plot(scalings = 'auto', start = 100, duration = 0.1, butterfly = True)
# raw.pick_types(meg = 'grad').plot(scalings = 'auto', start = 100, duration = 0.1, butterfly = True) # Warum gibt es keine channels vom Typ grad?

# Compare unfiltered/filtered power spectral density
raw.plot_psd()
raw_sss.plot_psd()

raw.plot(scalings = dict(mag = 8e-10, grad = 1e-9), clipping = 2, start = 100, duration = 0.5)
# raw.plot_sensors()

print("========END PLOTTING========\n")