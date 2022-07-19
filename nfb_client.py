# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:02:10 2022

This script starts a new Neurofeedback client that connects
to a streaming server (e.g. FieldTripBuffer or LSLServer, )

@author: Simon Kern
"""
import mne
from mne.preprocessing import maxwell_filter, find_bad_channels_maxwell
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
from mne_realtime import LSLClient, MockLSLStream
import matplotlib.pyplot as plt
import utils
import stimer
#%% SETTINGS

server = 'mock' # can be either FieldTripBuffer
fif_file = 'Z:/DeSMRRest/raw/dsmr_127/210908/RS2.fif' # a fif file that will be streamed by the simulator

crosstalk_file = './calibration_files/ct_sparse.fif'
fine_cal_file = './calibration_files/sss_cal.dat'

win_len = 1 # filtering window length in seconds


LSLClient.get_data_as_raw = utils.get_data_as_raw
#%% MAIN



if __name__=='__main__':
    
    raw_orig = mne.io.read_raw(fif_file, preload=True, verbose='WARNING')
    raw_orig.del_proj('all') # else they get copied to the layer, even though they are not applied
    
    with MockLSLStream('mock', raw_orig, 'all'):
        with LSLClient(info=raw_orig.info, host='mock', wait_max=10) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])

            # retrieve 
            for i in range(3):
                n_samples = int(win_len*sfreq)
                print('get data')
                raw = client.get_data_as_raw(n_samples=n_samples)
                print('finished data')

                stimer.start()
                # raw = utils.epoch2raw(epoch)
                auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(raw, 
                                                                          cross_talk = crosstalk_file, 
                                                                          calibration = fine_cal_file, 
                                                                          verbose = False)
                chpi_amplitudes = compute_chpi_amplitudes(raw, verbose = False)
                chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes, verbose = False)
                head_position = compute_head_pos(raw.info, chpi_locs, verbose = False)
                raw_sss = maxwell_filter(raw, int_order = 8, 
                                         ext_order = 3,
                                         head_pos = head_position, 
                                         calibration = fine_cal_file, 
                                         cross_talk = crosstalk_file, 
                                         mag_scale ='auto', 
                                         verbose = False)

                stimer.stop()
                