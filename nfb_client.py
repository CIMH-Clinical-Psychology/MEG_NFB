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
from mne.preprocessing import maxwell_filter, find_bad_channels_maxwell
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
from mne_realtime import LSLClient, MockLSLStream, FieldTripClient
import numpy as np
import matplotlib.pyplot as plt
import utils
import stimer
from joblib.memory import Memory
from multiprocessing import Process, Queue, shared_memory
from threading import Thread

# use caching for reading the raw file
mem = Memory(os.path.expanduser('~/mne-nfb/cache/')) 
mne.io.read_raw = mem.cache(mne.io.read_raw)


#%% MAIN

class HeadPosProcessor(Process):
    def __init__(self):
        super(HeadPosProcessor, self).__init__()
        
    def run(self):
        print('hello')

    def run_chunk(self):
        iterator = client.iter_raw_buffers()
        self.cliemnt
        while True:
            chpi_amplitudes = compute_chpi_amplitudes(raw, verbose = False)
            chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes, verbose = False)
            head_position = compute_head_pos(raw.info, chpi_locs, verbose = False)
            

      
class DataClient(Process):
    """Thread that continuously fetches data in 100ms chunks from the
    stream"""
    
    def _get_times(self):
        return np.ndarray(self.times.shape, dtype=self.times.dtype, buffer=self.times_shm.buf)
    
    def _get_buffer(self):
        return np.ndarray(self.buffer.shape, dtype=self.buffer.dtype, buffer=self.buffer_shm.buf)
    
    def __init__(self, client, buffersize=10):
        super(DataClient, self).__init__()
        self.client = client
        self.buffersize = buffersize
        self.sfreq = int(self.get_measurement_info('sfreq'))
        self.n_chs = self.get_measurement_info('nchan')
        self.pull_size = int(np.ceil(self.sfreq / 10))
        self.buffer = np.zeros([self.n_chs, self.sfreq*buffersize], dtype=np.float64)
        self.buffer_shm = shared_memory.SharedMemory(create=True, size=self.buffer.nbytes)
        self.times = np.zeros(buffersize*self.pull_size, dtype=np.float64)
        self.times_shm = shared_memory.SharedMemory(create=True, size=self.times.nbytes)
        # self.times = np.ndarray(times.shape, dtype=times.dtype, buffer=times_buff.buf)
        
    def run(self):
        print('run', flush=True)
        time.sleep(0.5)

        buffer = np.ndarray(self.buffer.shape, dtype=self.buffer.dtype, buffer=self.buffer_shm.buf)
        times = np.ndarray(self.times.shape, dtype=self.times.dtype, buffer=self.times_shm.buf)
        # times = self._get_times()
        print('loop', flush=True)
        
        # while True:
        #     # pulls continuously data and puts it in a rotating 
        #     # shared memory bufffer, i.e. the latest 100ms are always
        #     # stored at the end of the array
        #     print('loop')
        #     data, timestamp = self.client.client.pull_chunk(max_samples=self.pull_size)
        #     times[:-self.pull_size] = times[self.pull_size:]
        #     times[-self.pull_size:] = timestamp
        #     buffer[:, :-self.pull_size] = buffer[:, self.pull_size:]
        #     buffer[:, -self.pull_size:] = np.transpose(data)
        #     time.sleep(0.1)
            
        print('done', flush=True)
   
    def get_measurement_info(self, attr=None):
        if not hasattr(self, 'client_info'):
            self.client_info = self.client.get_measurement_info()
        return self.client_info if attr is None else self.client_info[attr]      
    
    
class LiveProcessor:
    
    def __init__(self, client, movement_corr=False, crosstalk_file=None,
                 fine_cal_file=None):
        self.client = client
        self.crosstalk_file = crosstalk_file
        self.fine_cal_file = fine_cal_file
        self.movement_corr = movement_corr
        self.data = []
                
        
    def start_loop(self):
        self.client_info = self.get_measurement_info()
        self.sfreq = int(self.client_info['sfreq'])
        # create shared memory buffer of 5 seconds
        self.buffer = shm.zeros(self.sfreq*5, dtype=np.float64)

        for i in range(5):
            self.filter_next()
        
    @stimer.wrapper
    def filter_next(self):
        # retrieve next set of data
    
        sfreq = int(self.get_measurement_info('sfreq'))
        n_samples = int(win_len*sfreq)

        raw = self.client.get_data_as_raw(n_samples=n_samples)
        
        if self.movement_corr:
            chpi_amplitudes = compute_chpi_amplitudes(raw, verbose = False)
            chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes, verbose = False)
            head_position = compute_head_pos(raw.info, chpi_locs, verbose = False)
        else:
            head_position = None
        
        raw_sss = maxwell_filter(raw, int_order = 8, 
                                 ext_order = 3,
                                 head_pos = head_position, 
                                 calibration = fine_cal_file, 
                                 cross_talk = crosstalk_file, 
                                 mag_scale ='auto', 
                                 verbose = False)
        self.data.append(raw_sss)
        return raw_sss
        
        
    def get_measurement_info(self, attr=None):
        if not hasattr(self, 'client_info'):
            self.client_info = self.client.get_measurement_info()
        return self.client_info if attr is None else self.client_info[attr]
    


if __name__=='__main__':
    pass
    #%% SETTINGS

    fif_file = utils.get_demo_rest_fif() # a fif file that will be streamed by the simulator

    crosstalk_file = './calibration_files/ct_sparse.fif'
    fine_cal_file = './calibration_files/sss_cal.dat'

    LSLClient.get_data_as_raw = utils.get_data_as_raw

    raw_orig = mne.io.read_raw(fif_file, preload=True, verbose='WARNING')
    raw_orig.del_proj('all') # else they get copied to the layer, even though they are not applied
    
    stream = MockLSLStream('mock', raw_orig, 'all', time_dilation=2).start()
    client = LSLClient(info=raw_orig.info, host='mock', wait_max=10).start()
    # client = FieldTripClient(info=raw_orig.info)

    # print('starting dataclient')
    dataclient = DataClient(client)
    # print('init')
    dataclient.start()
    print('now?')
    for i in range(5):
        print(i)
        print(type(dataclient.times))
        print(np.max(dataclient.times))
        time.sleep(2)
    # self = LiveProcessor(client, movement_corr=True)
    
