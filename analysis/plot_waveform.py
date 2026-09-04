import os
import re
import sys
import time
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import ds_style

fn = sys.argv[1]
rb = 1 if 2>=len(sys.argv) else int(sys.argv[2])
nwf = 20 if 3>=len(sys.argv) else int(sys.argv[3])

data = np.load(fn,allow_pickle=True)
for ikey in data.keys():
    print(ikey)
print('number of points =',data['npts'])
print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])

sample_rate = data['sample_rate']
npts = data['npts']
ref_position = data['ref_position']
tbin = (np.arange(npts) - npts*ref_position/100)/sample_rate
tbin = tbin.reshape(-1,rb).mean(axis=1)
nwf = data['ch1'].shape[0]

print(ref_position)
print(tbin)

ncol = 4
nrow = 5
fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=True)
for idx in range(nrow*ncol):
    irow = idx//ncol
    icol = idx%ncol
    if idx>=nwf:
        ax[irow,icol].axis('off')
        continue

    v0 = data['ch0'][idx].reshape(-1,rb).mean(axis=1)
    v1 = data['ch1'][idx].reshape(-1,rb).mean(axis=1)

    ax[irow,icol].plot(tbin,v0,'-',label='ch0',color='C0')
    ax[irow,icol].plot(tbin,v1,'-',label='ch1',color='C1')

    if irow==nrow-1:
        ax[irow,icol].set_xlabel('time [s]')
    if icol==0:
        ax[irow,icol].set_ylabel('voltage [V]')

for iax in ax.flatten():
    iax.grid()
    iax.legend(fontsize='small')

fig.tight_layout
plt.show()
