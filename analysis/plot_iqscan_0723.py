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
from scipy.optimize import least_squares
import datetime

import ds_style

freq = 5275

files = glob.glob(f'July23rd/iqscan0723/Scan1_{freq}MHz_m*db._calib.csv')
files = sorted(files, key=lambda f: float(re.search(r'MHz_m(\d+)db',f).group(1)))
dbs = [float(re.search(r'MHz_m(\d+)db',f).group(1)) for f in files]
dfs = [pd.read_csv(f, converters={"iq": complex, "iq_corr":complex}) for f in files]

files2 = glob.glob(f'July23rd/iqscan0723/Scan2_{freq}MHz_m*db._calib.csv')
files2 = sorted(files2, key=lambda f: float(re.search(r'MHz_m(\d+)db',f).group(1)))
dbs2 = [float(re.search(r'MHz_m(\d+)db',f).group(1)) for f in files2]
dfs2 = [pd.read_csv(f, converters={"iq": complex, "iq_corr":complex}) for f in files2]

print(dbs, dbs2)

fig,ax = plt.subplots(figsize=(16,9),ncols=2)
for idb,idf in zip(dbs,dfs):
    ax[0].plot(np.real(idf['iq']),np.imag(idf['iq']),'-o',label=f'-{idb} dB')
    ax[1].plot(np.real(idf['iq_corr']),np.imag(idf['iq_corr']),'-o',label=f'-{idb} dB')

ax[0].legend(title='Raw')
ax[1].legend(title='Calibrated')

for iax in ax:
    iax.grid()
    iax.set_xlabel('I')
    iax.set_ylabel('Q')
fig.tight_layout()
plt.show()

fig,ax = plt.subplots(figsize=(16,9),ncols=2)
for idb,idf in zip(dbs,dfs):
    ax[0].plot(idf['freq'],20*np.log10(np.abs(idf['iq_corr'])),'o-',label=f'-{idb} dB')
    ax[1].plot(idf['freq'],np.angle(idf['iq_corr']),'o-',label=f'-{idb} dB')

ax[0].set_ylabel('Amplitude [db]')
ax[1].set_ylabel('Phase')

for iax in ax:
    iax.grid()
    iax.legend()
    iax.set_xlabel('Frequency [GHz]')
fig.tight_layout()
plt.show()


for idb,idf2 in zip(dbs2,dfs2):
    print(dbs.index(idb))
    idf = dfs[dbs.index(idb)]
    fig,ax = plt.subplots(figsize=(16,9),ncols=2)

    ax[0].plot(np.real(idf['iq']),np.imag(idf['iq']),'-o',label=f'First')
    ax[0].plot(np.real(idf2['iq']),np.imag(idf2['iq']),'-o',label=f'Second')
    ax[1].plot(np.real(idf['iq_corr']),np.imag(idf['iq_corr']),'-o',label=f'First')
    ax[1].plot(np.real(idf2['iq_corr']),np.imag(idf2['iq_corr']),'-o',label=f'Second')

    for iax in ax:
        iax.grid()
        iax.set_xlabel('I')
        iax.set_ylabel('Q') 
    ax[0].legend(title=f'-{idb} dB, Raw')
    ax[1].legend(title=f'-{idb} dB, Calibrated')

    fig.tight_layout()
    plt.show()


    fig,ax = plt.subplots(figsize=(16,9),ncols=2)
    ax[0].plot(idf['freq'],20*np.log10(np.abs(idf['iq_corr'])),'o-',label=f'First')
    ax[0].plot(idf2['freq'],20*np.log10(np.abs(idf2['iq_corr'])),'o-',label=f'Second')
    ax[1].plot(idf['freq'],np.angle(idf['iq_corr']),'o-',label=f'First')
    ax[1].plot(idf2['freq'],np.angle(idf2['iq_corr']),'o-',label=f'Second')

    ax[0].set_ylabel('Amplitude [db]')
    ax[1].set_ylabel('Phase')

    for iax in ax:
        iax.grid()
        iax.legend()
        iax.set_xlabel('Frequency [GHz]')
    fig.tight_layout()
    plt.show()
