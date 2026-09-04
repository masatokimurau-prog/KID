
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
import sys

import os
listname = 'list_scan'
os.system(f'ls -rt wf_*.npz > {listname}')

with open(listname) as f:
    filenames = f.readlines()

fig, ax = plt.subplots(2,5, figsize=(18, 7))
ped0 = []
ped1 = []
ped_range = [-1000, -200]#ns
max0 = []
max1 = []
min0 = []
min1 = []
sum0 = []
sum1 = []
sum_range = [-200, 4000]#ns
sum_range = [-200, 1000]#ns
for filename in filenames:

    print(filename.strip())

    data = np.load(filename.strip())
    daq_rate = data['daq_rate']
    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    time = (np.arange(npts) - npts*ref_position/100)/sample_rate
    ch0 = -data['ch0']
    ch1 = data['ch1']
    nwf = ch0.shape[0]

    i_ped_range = np.zeros(2, dtype=int)
    i_sum_range = np.zeros(2, dtype=int)
    for jj in range(2):
        i_ped_range[jj] = np.argmin(np.abs(time*1e9 - ped_range[jj]))
        i_sum_range[jj] = np.argmin(np.abs(time*1e9 - sum_range[jj]))

    for ii in range(nwf):
        '''
        #aa.plot(time*1e6, ch0[ii], 'o', label=f'{ii}')
        #aa.plot(time*1e6, ch1[ii], 'o', label=f'{ii}')
        aa = ax[0]
        aa.plot(time*1e6, ch0[ii], label=f'{ii}')
        aa = ax[1]
        aa.plot(time*1e6, ch1[ii], label=f'{ii}')
        aa = ax[2]
        aa.plot(time*1e6, ch0[ii], label=f'{ii}')
        aa.plot(time*1e6, ch1[ii], label=f'{ii}')
        aa = ax[1]
        aa.plot(time[i_ped_range[0]]*1e6, 0.001, 'o')
        aa.plot(time[i_ped_range[1]]*1e6, 0.001, 'o')
        aa.plot(time[i_sum_range[1]]*1e6, 0.001, 'o')
        print(time[i_ped_range[0]]*1e6)
        print(time[i_ped_range[1]]*1e6)
        print(time[i_sum_range[0]]*1e6)
        print(time[i_sum_range[1]]*1e6)
        '''

        ped0tmp = ch0[ii][i_ped_range[0]:i_ped_range[1]].mean()
        ped1tmp = ch1[ii][i_ped_range[0]:i_ped_range[1]].mean()
        ped0.append(ped0tmp)
        ped1.append(ped1tmp)
        #sum0.append(ch0[ii][i_sum_range[0]:i_sum_range[1]].mean() - ped0tmp)
        #sum1.append(ch1[ii][i_sum_range[0]:i_sum_range[1]].mean() - ped1tmp)
        sum0.append(ch0[ii][i_sum_range[0]:i_sum_range[1]].mean() - ped0tmp)
        sum1.append(ch1[ii][i_sum_range[0]:i_sum_range[1]].mean() - ped1tmp)
        min0.append(ch0[ii].min())
        min1.append(ch1[ii].min())
        max0.append(ch0[ii].max() - ped0tmp)
        max1.append(ch1[ii].max() - ped1tmp)

ped0 = np.array(ped0)
ped1 = np.array(ped1)
max0 = np.array(max0)
max1 = np.array(max1)
min0 = np.array(min0)
min1 = np.array(min1)
sum0 = np.array(sum0)
sum1 = np.array(sum1)

nbin = 50
aa = ax[0, 0]
aa.hist(ped0, bins = nbin)
aa = ax[0, 1]
#aa.hist(min0, bins = nbin)
aa.plot(np.arange(len(ped0)), ped0)
aa = ax[0, 2]
aa.hist(max0, bins = nbin)
aa = ax[0, 3]
aa.hist(sum0, bins = nbin)

aa = ax[1, 0]
aa.hist(ped1, bins = nbin)
aa = ax[1, 1]
#aa.hist(min1, bins = nbin)
aa.plot(np.arange(len(ped0)), ped1)
aa = ax[1, 2]
aa.hist(max1, bins = nbin)
aa = ax[1, 3]
aa.hist(sum1, bins = nbin)

aa = ax[0, 4]
aa.hist2d(ped0, max1, bins = nbin, norm=matplotlib.colors.LogNorm(), cmap = cm.jet)
#aa.hist2d(ped0, max1, bins = 30, range = [[0.035, 0.05], [0, 0.02]], norm=matplotlib.colors.LogNorm(), cmap = cm.jet)

aa = ax[1, 4]
aa.hist2d(ped0, sum1, bins = nbin, norm=matplotlib.colors.LogNorm(), cmap = cm.jet)
#aa.hist2d(ped0, sum1, bins = 30, range = [[0.035, 0.05], [0, 0.01]], norm=matplotlib.colors.LogNorm(), cmap = cm.jet)
#aa.hist2d(max0, max1, bins = nbin, cmap = cm.jet)


'''
aa.set_xlabel(r'Time ($\mu$s)')
aa.set_ylabel('Voltage (V)')
aa = ax[1]
aa.set_xlabel(r'Time ($\mu$s)')
aa = ax[2]
aa.set_xlabel(r'Time ($\mu$s)')
'''

fig.tight_layout()
#fig.savefig('aa.svg')
plt.show()
