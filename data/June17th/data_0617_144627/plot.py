
import matplotlib.pyplot as plt
import numpy as np

import os
listname = 'list_scan'
os.system(f'ls -rt wf_*.npz > {listname}')

with open(listname) as f:
    filenames = f.readlines()

fig, ax = plt.subplots(1,3, figsize=(9, 5))
for filename in filenames:

    data = np.load(filename.strip())
    daq_rate = data['daq_rate']
    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    time = (np.arange(npts) - npts*ref_position/100)/sample_rate
    ch0 = data['ch0']
    ch1 = data['ch1']
    nwf = ch0.shape[0]

    for ii in range(nwf):
        #aa.plot(time*1e6, ch0[ii], 'o', label=f'{ii}')
        #aa.plot(time*1e6, ch1[ii], 'o', label=f'{ii}')
        aa = ax[0]
        aa.plot(time*1e6, ch0[ii], label=f'{ii}')
        aa = ax[1]
        aa.plot(time*1e6, ch1[ii], label=f'{ii}')
        aa = ax[2]
        aa.plot(time*1e6, ch0[ii], label=f'{ii}')
        aa.plot(time*1e6, ch1[ii], label=f'{ii}')

if len(filenames)*nwf < 11:
    aa.legend()

aa = ax[0]
aa.set_xlabel(r'Time ($\mu$s)')
aa.set_ylabel('Voltage (V)')
aa = ax[1]
aa.set_xlabel(r'Time ($\mu$s)')
aa = ax[2]
aa.set_xlabel(r'Time ($\mu$s)')

fig.tight_layout()
fig.savefig('aa.svg')
plt.show()
