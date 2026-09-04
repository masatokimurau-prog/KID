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

def Table_Temp(kelvin):
    if 4.76==kelvin: 
        return {0: 'July3rd/wf_260703_143924_49.89Hz.npz', -30: 'July3rd/wf_260703_151037_49.92Hz.npz', +30: 'July3rd/wf_260703_151317_49.86Hz.npz'}
    if 5.29==kelvin: 
        return {0: 'July3rd/wf_260703_152553_49.83Hz.npz', -30: 'July3rd/wf_260703_152642_49.90Hz.npz', +30: 'July3rd/wf_260703_152732_49.87Hz.npz'}
    if 5.80==kelvin: 
        return {0: 'July3rd/wf_260703_154738_49.75Hz.npz', -30: 'July3rd/wf_260703_154830_49.83Hz.npz', +30: 'July3rd/wf_260703_154921_49.79Hz.npz'}
    if 6.30==kelvin: 
        return {0: 'July3rd/wf_260703_160556_49.88Hz.npz', -30: 'July3rd/wf_260703_160649_49.91Hz.npz', +30: 'July3rd/wf_260703_160750_49.93Hz.npz'}
    if 7.30==kelvin: 
        return {0: 'July3rd/wf_260703_162311_49.84Hz.npz', -30: 'July3rd/wf_260703_162401_49.85Hz.npz', +30: 'July3rd/wf_260703_162451_49.90Hz.npz'}



##### Calibration
fcalib = sys.argv[1]
kelvin = float(re.search(r'(\d+\.\d+)K',fcalib).group(1))
data = np.load(fcalib, allow_pickle=True)['dd']

df = pd.DataFrame(data,columns=['freq','ch0','ch1'])
df['freq'] = df['freq'].astype(float)/1e9
for ich in ['ch0','ch1']:
    df[ich] = df[ich].astype(float)*1e3
df['iq'] = df['ch0'] + 1j*df['ch1']
print(df)
ll = len(df)

fr_idx = np.argmin(np.abs(df['iq']))
fr = df['freq'][fr_idx]

a1,b1 = np.polyfit(df['freq'].values[0:3],np.angle(df['iq'])[0:3],1)
a2,b2 = np.polyfit(df['freq'].values[-6:-3],np.angle(df['iq'])[-6:-3],1)
tau = (a1+a2)/2/(2*np.pi)
alpha = np.angle(df['iq'][fr_idx]) - 2*np.pi*df['freq'][fr_idx]*tau

df['iq_corr'] = df['iq']*np.exp(-1j*(2*np.pi*df['freq']*tau + alpha))

circle_res = least_squares(funcCircle, x0=[np.max(np.abs(df['iq_corr']))/2, 0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-10:fr_idx+10]),np.imag(df['iq_corr'][fr_idx-10:fr_idx+10])))
# circle_res = least_squares(funcCircle_xaxis, x0=[0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-10:fr_idx+10]),np.imag(df['iq_corr'][fr_idx-10:fr_idx+10])))
# circle_res.x = [circle_res.x[0], 0, circle_res.x[1]]
print(circle_res)
print(fr, tau, alpha)

cp = {0: 'blue', fr_idx: 'red', ll-1: 'green'}

fig,ax = plt.subplots(figsize=(16,9),ncols=3,nrows=2)

iax = ax[0,0]
iax.plot(df['ch0'],df['ch1'],'-o',label='iq')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['ch0'][idx],df['ch1'][idx],'o',ms=10,color=cp[idx])
iax.set_xlabel('ch0 [mV]')
iax.set_ylabel('ch1 [mV]')

iax = ax[0,1]
ax_r = iax.twinx()
iax.plot(df['freq'],np.abs(df['iq']),'-o',label='abs')
ax_r.plot(df['freq'],20*np.log10(np.abs(df['iq'])),'-o',color='C4')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.abs(df['iq'][idx]),'o',ms=10,color=cp[idx])
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('magnitude [mV]')
ax_r.set_ylabel('magnitude [dB]')
ax_r.yaxis.label.set_color('C4')

iax = ax[0,2]
iax.plot(df['freq'],np.angle(df['iq']),'-o',label='angle')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.angle(df['iq'][idx]),'o',ms=10,color=cp[idx])
iax.plot(df['freq'],np.angle(funcPhase(df['freq'], tau, alpha)),'-',label='env term')
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('phase [rad]')

# iax = ax[1,0]
# iax.plot(df['freq'],np.real(df['iq']),'-o',label='real')
# iax.plot(df['freq'],np.imag(df['iq']),'-o',label='imag')
# iax.set_xlabel('frequency [GHz]')
# iax.set_ylabel('[mV]')

iax = ax[1,0]
iax.plot(np.real(df['iq_corr']),np.imag(df['iq_corr']),'-o',label='iq')
for idx in [0,fr_idx,ll-1]:
    iax.plot(np.real(df['iq_corr'])[idx],np.imag(df['iq_corr'])[idx],'o',ms=10,color=cp[idx])
iax.plot(circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)), circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)),'-',label='circle fit')
iax.plot(circle_res.x[0],circle_res.x[1],'*',ms=10,color='C3',label='center')
iax.set_xlabel('I [mV]')
iax.set_ylabel('Q [mV]')

iax = ax[1,1]
iax.plot(df['freq'],np.abs(df['iq_corr']),'-o',label='abs')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.abs(df['iq_corr'][idx]),'o',ms=10,color=cp[idx])
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('magnitude [mV]')

iax = ax[1,2]
iax.plot(df['freq'],np.angle(df['iq_corr']),'-o',label='angle')
for idx in [0,fr_idx,ll-1]:
    iax.plot(df['freq'][idx],np.angle(df['iq_corr'][idx]),'o',ms=10,color=cp[idx])
iax.set_xlabel('frequency [GHz]')
iax.set_ylabel('phase [rad]')


for iax in ax.flatten():
    iax.grid()
    iax.legend(fontsize='small')

fig.tight_layout()
plt.show()


##### Apply calibration to data
npzfiles = Table_Temp(kelvin)

fdata = npzfiles[0.0].replace('.npz', '_res.npz')
if not 'res.npz' in fdata:
    print('Please provide the result file from ana_iqscan_July3rd.py')
    sys.exit()
res = np.load(fdata)

ped_data = res['ch0_ped']+1j*res['ch1_ped']
ped_data_corr = ped_data*np.exp(-1j*(2*np.pi*fr*tau + alpha))
norm0 = np.hypot(circle_res.x[0],circle_res.x[1])
norm1 = np.hypot(circle_res.x[0] + circle_res.x[2]*circle_res.x[0]/norm0, circle_res.x[1] + circle_res.x[2]*circle_res.x[1]/norm0)
norm2 = norm1-norm0
ped_data_norm = ped_data_corr/norm1*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))

peakv_ch = [np.where(res[f'ch{ich}_integ200']>0, res[f'ch{ich}_peak_max'], res[f'ch{ich}_peak_min']) for ich in range(2)]
peakv = peakv_ch[0] + 1j*peakv_ch[1]
peakv_corr = peakv*np.exp(-1j*(2*np.pi*fr*tau + alpha))
peakv_norm = peakv_corr/norm1*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))

fig2,ax2 = plt.subplots(figsize=(16,9),ncols=3,nrows=2)

for idx in range(2):
    iax = ax2[idx,0]
    iax.plot(res['ch0_ped'],res['ch1_ped'],'.',alpha=0.1, label='Pedestal')
    # iax.plot(peakv_ch[0],peakv_ch[1],'.',alpha=0.1, label='Peak')
    iax.set_xlabel('ch0 [mV]')
    iax.set_ylabel('ch1 [mV]')

    iax = ax2[idx,1]
    iax.plot(np.real(ped_data_corr), np.imag(ped_data_corr), '.', alpha=0.1, label='Pedestal')
    # iax.plot(np.real(peakv_corr), np.imag(peakv_corr), '.', alpha=0.1, label='Peak')
    circle_curve = circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)) + 1j*(circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)))
    # iax.plot(circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)), circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)),'-',label='circle fit')
    iax.plot(np.real(circle_curve), np.imag(circle_curve),'-',label='circle fit')
    iax.plot(circle_res.x[0],circle_res.x[1],'*',ms=10,color='C3',label='center')
    iax.set_xlabel('I [mV]')
    iax.set_ylabel('Q [mV]')

    iax = ax2[idx,2]
    iax.plot(np.real(ped_data_norm), np.imag(ped_data_norm), '.', alpha=0.1, label='Pedestal')
    zc = circle_res.x[0]*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))/norm1 + 1j*circle_res.x[1]*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))/norm1
    circle_curve_norm = (circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)))*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))/norm1 + 1j*(circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)))*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))/norm1
    iax.plot(np.real(circle_curve_norm), np.imag(circle_curve_norm),'-',color='k')
    iax.plot(zc.real, zc.imag,'*',ms=10,color='C3',label='center')
    # iax.plot(np.real(peakv_norm), np.imag(peakv_norm), '.', alpha=0.1, label='Peak')
    iax.set_xlabel('I (normalized)')
    iax.set_ylabel('Q (normalized)')

for iax in ax2.flatten():
    iax.grid()
    iax.legend(fontsize='small')

for idx in range(2):
    ax2[idx,2].set_xlim(-1,1)
    ax2[idx,2].set_ylim(-1,1)

# fig2.tight_layout()
# plt.show()


##### Apply calibration to trajectory
if len(sys.argv)<=2: sys.exit()
nplot =  int(sys.argv[2])

for idx,(idf,ifile) in enumerate(npzfiles.items()):
    rawdata = np.load(ifile,allow_pickle=True)
    # if 0<idx: continue

    sample_rate = rawdata['sample_rate']
    npts = rawdata['npts']
    ref_position = rawdata['ref_position']
    nwf = rawdata['ch1'].shape[0]

    todraw = rawdata['ch0']*1e3 + 1j*rawdata['ch1']*1e3
    todraw_corr = todraw*np.exp(-1j*(2*np.pi*(fr+idf*1e-3)*tau + alpha))
    todraw_norm = todraw_corr/norm1*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0])) 
    print(todraw_norm.shape)

    for jplot in range(nplot):
        iplot = jplot*10
        print(todraw[iplot].shape)
        dopt = {'alpha': 0.1, 'color': plt.cm.viridis(idx/len(npzfiles)), 'lw': 1}
        if 0==iplot:
            dopt['label'] = f'{idf} MHz'
        iax = ax2[1,0]
        iax.plot(np.real(todraw[iplot][::2]), np.imag(todraw[iplot][::2]), '-', **dopt)
        iax = ax2[1,1]
        iax.plot(np.real(todraw_corr[iplot][::2]), np.imag(todraw_corr[iplot][::2]), '-', **dopt)
        iax = ax2[1,2]
        iax.plot(np.real(todraw_norm[iplot][::2]), np.imag(todraw_norm[iplot][::2]), '-', **dopt)
    

fig2.tight_layout()
plt.show()

sys.exit()

fig3,ax3 = plt.subplots(figsize=(16,9),ncols=int(np.ceil(np.sqrt(nplot))),nrows=int(np.ceil(nplot/np.ceil(np.sqrt(nplot)))),sharex=True,sharey=True)
for idx in range(nplot):
    irow = idx//int(np.ceil(np.sqrt(nplot)))
    icol = idx%int(np.ceil(np.sqrt(nplot)))
    if idx>=nwf:
        ax3[irow,icol].axis('off')
        continue

    iax = ax3[irow,icol]

    iax.plot(np.real(todraw_norm[idx]), np.imag(todraw_norm[idx]), '.', alpha=0.1, label='norm')
    iax.set_xlabel('real [mV]')
    iax.set_ylabel('imag [mV]')





