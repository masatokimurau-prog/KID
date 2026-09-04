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

def funcPhase(f, tau, alpha):
    return np.exp(1j*(2*np.pi*f*tau + alpha))

def funcCircle(pars, x, y,):
    xc, yc, r = pars
    return (x-xc)**2 + (y-yc)**2 - r**2

def funcCircle_xaxis(pars, x, y,):
    xc, r = pars
    yc = 0
    return (x-xc)**2 + (y-yc)**2 - r**2

fn = sys.argv[1]

data = np.load(fn, allow_pickle=True)['dd']
df = pd.DataFrame(data,columns=['freq','ch0','ch1'])
df['freq'] = df['freq'].astype(float)/1e9
for ich in ['ch0','ch1']:
    df[ich] = df[ich].astype(float)*1e3
# df['iq'] = df['ch0'] + 1j*df['ch1']
df['iq'] = 1j*df['ch0'] + df['ch1']
print(df)
ll = len(df)

fr_idx = np.argmin(np.abs(df['iq']))
fr = df['freq'][fr_idx]

a1,b1 = np.polyfit(df['freq'].values[0:3],np.angle(df['iq'])[0:3],1)
a2,b2 = np.polyfit(df['freq'].values[-6:-3],np.angle(df['iq'])[-6:-3],1)
tau = (a1+a2)/2/(2*np.pi)
alpha = np.angle(df['iq'][fr_idx]) - 2*np.pi*df['freq'][fr_idx]*tau

df['iq_corr'] = df['iq']*np.exp(-1j*(2*np.pi*df['freq']*tau + alpha))

circle_res = least_squares(funcCircle, x0=[np.max(np.abs(df['iq_corr']))/2, 0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-5:fr_idx+5]),np.imag(df['iq_corr'][fr_idx-5:fr_idx+5])))

norm0 = np.hypot(circle_res.x[0],circle_res.x[1])
norm1 = np.hypot(circle_res.x[0] + circle_res.x[2]*circle_res.x[0]/norm0, circle_res.x[1] + circle_res.x[2]*circle_res.x[1]/norm0)
norm2 = norm1-norm0

df.to_csv(fn.replace('npz','_calib.csv'),index=False)

# cp = {0: 'blue', fr_idx: 'red', ll-1: 'green'}
cp = {0: 'k', fr_idx: 'k', ll-1: 'k'}
mm = {0: '>', fr_idx: '*', ll-1: '<'}

######
fig,ax = plt.subplots(figsize=(8,7))
ax.plot(df['ch0'],df['ch1'],'-o',ms=5,label='iq',color='k')
for idx in [0,fr_idx,ll-1]:
    ax.plot(df['ch0'][idx],df['ch1'][idx],mm[idx],ms=15,color=cp[idx])
ax.set_xlabel('ch0 [mV]')
ax.set_ylabel('ch1 [mV]')
ax.grid()
plt.show()

######
fig,ax = plt.subplots(figsize=(8,7))
ax.plot(np.real(df['iq_corr']),np.imag(df['iq_corr']),'-o',ms=5,color='k',label='iq')
for idx in [0,fr_idx,ll-1]:
    ax.plot(np.real(df['iq_corr'])[idx],np.imag(df['iq_corr'])[idx],mm[idx],ms=15,color=cp[idx])
ax.plot(circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)), circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)),'--',color='gray',label='circle fit')
ax.plot(circle_res.x[0],circle_res.x[1],'x',ms=15,color='k',label='center')
ax.set_xlabel('I [mV]')
ax.set_ylabel('Q [mV]')
ax.legend()
ax.grid()
plt.show()

######
fig,ax = plt.subplots(figsize=(16,7),ncols=2)
iax = ax[0]
iax.plot(df['freq'],np.abs(df['iq_corr']),'-o',ms=5,color='k',label='abs')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.abs(df['iq_corr'][idx]),mm[idx],ms=15,color=cp[idx])
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('magnitude [mV]')

iax = ax[1]
iax.plot(df['freq'],np.angle(df['iq_corr']),'-o',ms=5,color='k',label='angle')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.angle(df['iq_corr'][idx]),mm[idx],ms=15,color=cp[idx])
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('phase [rad]')

for iax in ax:
    iax.grid()
fig.tight_layout()
plt.show()