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

import ds_style

def funcPhase(f, tau, alpha):
    return np.exp(1j*(2*np.pi*f*tau + alpha))

def funcCircle(pars, x, y,):
    xc, yc, r = pars
    return (x-xc)**2 + (y-yc)**2 - r**2

def funcCircle_xaxis(pars, x, y,):
    xc, r = pars
    yc = 0
    return (x-xc)**2 + (y-yc)**2 - r**2


files = glob.glob('July3rd/iqscan0703/*.npz')
files = [f for f in files if 'wide' not in f]
files = sorted(files, key=lambda x: float(re.search(r'(\d+\.\d+)K',x).group(1)))[:5]
kelvins = [float(re.search(r'(\d+\.\d+)K',f).group(1)) for f in files]

dfs = {}
fitres = []

for file, kelvin in zip(files, kelvins):
    data = np.load(file, allow_pickle=True)['dd']
    df = pd.DataFrame(data,columns=['freq','ch0','ch1'])
    df['freq'] = df['freq'].astype(float)/1e9
    for ich in ['ch0','ch1']:
        df[ich] = df[ich].astype(float)*1e3
    df['iq'] = df['ch0'] + 1j*df['ch1']

    ll = len(df)

    fr_idx = np.argmin(np.abs(df['iq']))
    fr = df['freq'][fr_idx]

    a1,b1 = np.polyfit(df['freq'].values[0:3],np.angle(df['iq'])[0:3],1)
    a2,b2 = np.polyfit(df['freq'].values[-6:-3],np.angle(df['iq'])[-6:-3],1)
    tau = (a1+a2)/2/(2*np.pi)
    alpha = np.angle(df['iq'][fr_idx]) - 2*np.pi*df['freq'][fr_idx]*tau

    df['iq_corr'] = df['iq']*np.exp(-1j*(2*np.pi*df['freq']*tau + alpha))

    circle_res = least_squares(funcCircle, x0=[np.max(np.abs(df['iq_corr']))/2, 0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-10:fr_idx+10]),np.imag(df['iq_corr'][fr_idx-10:fr_idx+10])))

    dfs[kelvin] = df
    fitres.append({'kelvin': kelvin, 'tau': tau, 'alpha': alpha, 'xc': circle_res.x[0], 'yc': circle_res.x[1], 'r': circle_res.x[2]})

dffit = pd.DataFrame(fitres)

fig,ax = plt.subplots(figsize=(16,9), ncols=3,nrows=2)

iax = ax[0,0]
for kelvin, df in dfs.items():
    iax.plot(df['ch0'], df['ch1'], '.', alpha=0.8, label=f'{kelvin}K')
iax.set_xlabel('ch0 [mV]')
iax.set_ylabel('ch1 [mV]')

iax = ax[0,1]
for kelvin, df in dfs.items():
    iax.plot(np.real(df['iq_corr']), np.imag(df['iq_corr']), '.', alpha=0.8, label=f'{kelvin}K')
iax.set_xlabel('I [mV]')
iax.set_ylabel('Q [mV]')

iax = ax[1,0]
ax_r = iax.twinx()
iax.plot(dffit['kelvin'], dffit['tau']*1e9, '-o')
ax_r.plot(dffit['kelvin'], dffit['alpha'], '-o', color='C1')
ax_r.tick_params(axis='y', colors='C1')
iax.set_xlabel('Temperature [K]')
iax.set_ylabel('tau [ns]')
ax_r.set_ylabel('alpha [rad]', color='C1')

for iax in ax.flatten():
    iax.grid()
    iax.legend(fontsize='small',ncol=2)

fig.tight_layout()
plt.show()