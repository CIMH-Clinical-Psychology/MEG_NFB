"""
        Accessing FieldTrip buffer with python
"""

import sys
from fieldtrip import FieldTrip

ftc = FieldTrip.Client()
ftc.connect('localhost', 1972)

while True:
    
    h = ftc.getHeader()
    
    if h is None:
        print('Failed to retrieve header!')
        sys.exit(1)
    
    print(h)
    print(h.labels)
    
    if h.nSamples > 0:
      print('Trying to read last sample...')
      index = h.nSamples - 1
      D = ftc.getData([index, index])
      print(D)
    
    if h.nEvents > 0:
        print('Trying to read (all) events...')
        E = ftc.getEvents()
    
        for e in E:
            print(e)

# ftc.disconnect()