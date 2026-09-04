
import matplotlib.pyplot as plt
import numpy as np
import sys
if len(sys.argv)<4:
    print('give i_file, i_wf, and channel')
    sys.exit()
ifile = int(sys.argv[1])
iwf = int(sys.argv[2])
ich = int(sys.argv[3])

import os
listname = 'list_scan'
os.system(f'ls -rt wf_*.npz > {listname}')

with open(listname) as f:
    filenames = f.readlines()

fig, ax = plt.subplots(4,5, figsize=(15, 7))
jj = 0
for filename in filenames:

    if jj != ifile:
        jj += 1
        continue

    data = np.load(filename.strip())
    daq_rate = data['daq_rate']
    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    time = (np.arange(npts) - npts*ref_position/100)/sample_rate
    ch0 = data['ch0'][iwf:]
    ch1 = data['ch1'][iwf:]
    nwf = ch0.shape[0]
    nwf = 20

    for ii in range(nwf):
        mm = ch1[ii].max()
        color = 'k'
        if mm>0.01:
            color = 'r'
        if mm<0.002:
            color = 'b'
        
        aa = ax[ii//5, ii%5]
        if ich == 0:
            aa.plot(time*1e9, ch0[ii]*1e3, color+'-', label=f'{ii}')
        elif ich == 1:
            aa.plot(time*1e9, ch1[ii]*1e3, color+'-', label=f'{ii}')
        elif ich == 2:
            aa.plot(time*1e9, ch0[ii]*1e3, 'r-', label=f'{ii}')
            aa.plot(time*1e9, ch1[ii]*1e3, 'b-', label=f'{ii}')
        #aa.set_xlim(-100, 500)

    jj += 1
'''
aa = ax[0]
aa.set_xlabel(r'Time ($\mu$s)')
aa.set_ylabel('Voltage (V)')
aa = ax[1]
aa.set_xlabel(r'Time ($\mu$s)')
aa = ax[2]
aa.set_xlabel(r'Time ($\mu$s)')
'''

fig.tight_layout()
fig.savefig('aa.svg')
plt.show()
