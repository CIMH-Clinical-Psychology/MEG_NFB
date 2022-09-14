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


# use caching for reading the raw file
mem = Memory(os.path.expanduser('~/mne-nfb/cache/')) 
mne.io.read_raw = mem.cache(mne.io.read_raw)


#%% MAIN
if __name__=='__main__':
    
    # SETTINGS

    fif_file = utils.get_demo_rest_fif() # a fif file that will be streamed by the simulator

    crosstalk_file = './calibration_files/ct_sparse.fif'
    fine_cal_file = './calibration_files/sss_cal.dat'

    LSLClient.get_data_as_raw = utils.get_data_as_raw

    raw_orig = mne.io.read_raw(fif_file, preload=True, verbose='WARNING')
    raw_orig.del_proj('all') # else they get copied to the layer, even though they are not applied
    
    # stream = MockLSLStream('mock', raw_orig, 'all', time_dilation=2).start()
    # client = LSLClient(info=raw_orig.info, host='mock', wait_max=10).start()
    # client = FieldTripClient('localhost')
    # start a FieldTripBuffer Simulator. It will feed 1000 samples/second.
    client = FieldTripClientSimulator()
    
    # start the data client that is the central hub of retrieving
    # and storing the data from the MEG into our shared memory
    dataclient = MaxFilterCoordinator(client)
    
    dataclient.start()
    
    plt.figure()
    for i in range(10):
        print('main running...')
        # print(type(dataclient.times))
        with dataclient.lock:
            data = dataclient._buffer
        plt.plot(data[0])
        
        # plt.show(block=False)
        plt.pause(1)
    # self = LiveProcessor(client, movement_corr=True)
    
