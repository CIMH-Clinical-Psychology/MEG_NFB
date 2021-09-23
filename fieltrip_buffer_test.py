"""
        Accessing FieldTrip buffer with python while running FieldTrip fileproxy with pre-recorded data
        FieldTrip fileproxy is running in MATLAB
        Buffer = localhost:1972
"""

import sys
from fieldtrip import FieldTrip

# Creating and connecting client to FieldTrip buffer
ftc = FieldTrip.Client()
ftc.connect('localhost', 1972)

# Read 100 headers and data frames
for i in range(100):
    
    # Retrieve header of latest chunk in buffer
    header = ftc.getHeader()
    
    # Check for errors
    if header is None:
        print('Failed to retrieve header!')
        sys.exit(1)
    
    # Print general header information (number of channels, sample frequency, etc.)
    print(header)

    # Print labels of every channel
    print(header.labels)
    
    # Read data from last sample
    if header.nSamples > 0:
      print('Trying to read last sample...')
      index = header.nSamples - 1
      data = ftc.getData([index, index])
      
      # Print data as array of individual sensor signals
      print(data)
    
    # Read events --> irrelevant for our real-time application?
    if header.nEvents > 0:
        print('Trying to read (all) events...')
        E = ftc.getEvents()
    
        for e in E:
            print(e)

# Disconnect client from FieldTrip buffer. This crashes the MATLAB instance running the buffer/fileproxy for unknown reasons.
ftc.disconnect()
