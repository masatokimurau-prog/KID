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


def ProcessData(fn_):
    data = np.load(fn_,allow_pickle=True)
    for ikey in data.keys():
        print(ikey)
    print('number of points =',data['npts'])
    print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])

    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    nwf = data['ch1'].shape[0]

    print(ref_position)
    print(sample_rate)
    print(data['deltat'])

    pedbin = [0,int(npts*ref_position/100)]
    bin200 = pedbin[1] + int(200e-9*sample_rate)
    bin400 = pedbin[1] + int(400e-9*sample_rate)
    bin600 = pedbin[1] + int(600e-9*sample_rate)
    bin800 = pedbin[1] + int(800e-9*sample_rate)
    bin1000 = pedbin[1] + int(1000e-9*sample_rate)
    print(pedbin,bin200,bin400,bin800,bin1000)
    dres = {}
    for ich in range(2):
        ped = data[f'ch{ich}'][:,pedbin[0]:pedbin[1]].mean(axis=1)
        peak_max = data[f'ch{ich}'][:,pedbin[1]:].max(axis=1)
        peak_min = data[f'ch{ich}'][:,pedbin[1]:].min(axis=1)

        integ200 = data[f'ch{ich}'][:,pedbin[1]:bin200].sum(axis=1) - ped*(bin200-pedbin[1])
        integ400 = data[f'ch{ich}'][:,bin200:bin400].sum(axis=1) - ped*(bin400-bin200)
        integ600 = data[f'ch{ich}'][:,bin400:bin600].sum(axis=1) - ped*(bin600-bin400)
        integ800 = data[f'ch{ich}'][:,bin600:bin800].sum(axis=1) - ped*(bin800-bin600)
        integ1000 = data[f'ch{ich}'][:,bin800:bin1000].sum(axis=1) - ped*(bin1000-bin800)

        dres[f'ch{ich}_ped'] = ped*1e3
        dres[f'ch{ich}_peak_max'] = peak_max*1e3
        dres[f'ch{ich}_peak_min'] = peak_min*1e3
        dres[f'ch{ich}_integ200'] = integ200*1e3
        dres[f'ch{ich}_integ400'] = integ400*1e3
        dres[f'ch{ich}_integ600'] = integ600*1e3
        dres[f'ch{ich}_integ800'] = integ800*1e3
        dres[f'ch{ich}_integ1000'] = integ1000*1e3

    rebin = 10
    v0 = data['ch0'].reshape(nwf,-1,rebin).mean(axis=2)
    v1 = data['ch1'].reshape(nwf,-1,rebin).mean(axis=2)


    dres['deltat'] = data['deltat']
    dres['ch0'] = v0*1e3
    dres['ch1'] = v1*1e3
    # df = pd.DataFrame(dres)
    # print(df)
    # df.to_csv(fn_.replace('.npz','.csv'),index=False)
    # np.savez(fn_.replace('.npz','_res.npz'),**dres)
    return dres


fcalib = sys.argv[1]
ftemp = sys.argv[2]
ldatadir = sys.argv[3]
ld_freq = float(sys.argv[4])

##### Calibration
cdata = np.load(fcalib, allow_pickle=True)['dd']

df = pd.DataFrame(cdata,columns=['freq','ch0','ch1'])
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

circle_res = least_squares(funcCircle, x0=[np.max(np.abs(df['iq_corr']))/2, 0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-5:fr_idx+5]),np.imag(df['iq_corr'][fr_idx-5:fr_idx+5])))
# circle_res = least_squares(funcCircle_xaxis, x0=[0, np.max(np.abs(df['iq_corr']))], args=(np.real(df['iq_corr'][fr_idx-10:fr_idx+10]),np.imag(df['iq_corr'][fr_idx-10:fr_idx+10])))
# circle_res.x = [circle_res.x[0], 0, circle_res.x[1]]
print(circle_res)

norm0 = np.hypot(circle_res.x[0],circle_res.x[1])
norm1 = np.hypot(circle_res.x[0] + circle_res.x[2]*circle_res.x[0]/norm0, circle_res.x[1] + circle_res.x[2]*circle_res.x[1]/norm0)
norm2 = norm1-norm0


##### Temperature
with open(ftemp) as f:
    next(f)
    temp_start = next(f).rstrip()
temp_start = temp_start.split(',')[1]
temp_start = datetime.datetime.strptime(temp_start, "%a %b %d %H:%M:%S JST %Y")
print(temp_start)
dfTemp = pd.read_csv(ftemp,skiprows=3)
dfTemp['Date'] = temp_start+pd.to_timedelta(dfTemp['Time'], unit='ms')
dfTemp = dfTemp.set_index('Date')
print(dfTemp)

##### Laser data
lfiles = glob.glob(ldatadir+'/wf*.npz')
lfiles = sorted(lfiles, key=lambda x: int(re.search(r"wf_\d+_(\d+)_",x).group(1)))
print(lfiles)

liq = []
lts = []
ltod = []
for ifile in lfiles:
    dres = ProcessData(ifile)
    pedestal = dres['ch0_ped'] + 1j*dres['ch1_ped']
    iq = pedestal*np.exp(-1j*(2*np.pi*ld_freq*tau + alpha))
    liq.append(iq)

    t0 = re.search(r"wf_\d+_(\d+)_",ifile).group(1)
    t0 = datetime.datetime.strptime(t0, "%H%M%S")
    tday = re.search(r"wf_(\d+)_",ifile).group(1)
    t0 = t0.replace(year=2000+int(tday[0:2]), month=int(tday[2:4]), day=int(tday[4:6]))
    ts = t0 + dres['deltat']
    lts.append(ts)

    tod = dres['ch0'] + 1j*dres['ch1']
    tod = tod*np.exp(-1j*(2*np.pi*ld_freq*tau + alpha))
    ltod.append(tod)

liq = np.concatenate(liq)
lts = np.concatenate(lts)
ldt = np.array([(t - lts[0]).total_seconds() for t in lts])
print(ldt)
ltod = np.concatenate(ltod)

t0 = lts[0]
t1 = lts[-1]
print('start =', t0)
print('stop  =', t1)

plot_td = 5
sel_time = np.array([t0 + i*datetime.timedelta(seconds=plot_td) for i in range(int((t1-t0).total_seconds()/plot_td)+1)])
sel_idx = np.searchsorted(lts, sel_time)

liq = liq[sel_idx]
lts = lts[sel_idx]
ldt = ldt[sel_idx]
ltod = ltod[sel_idx]
ltemp = np.interp(pd.to_datetime(lts), dfTemp.index.astype(np.int64).to_numpy(), dfTemp["Input 1"].to_numpy())

liq_norm = liq/norm1*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))
ltod_norm = ltod/norm1*np.exp(-1j*np.arctan2(circle_res.x[1],circle_res.x[0]))

# print(lts)

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


fig2,ax2 = plt.subplots(figsize=(12,9),ncols=2,nrows=2)

iax = ax2[0,0]
im = iax.scatter(np.real(liq),np.imag(liq),c=ltemp,cmap='jet')
iax.set_xlabel('Pedestal I [mV]')
iax.set_ylabel('Pedestal Q [mV]')

iax = ax2[0,1]
for idx,itod in enumerate(ltod):
    c = im.cmap(im.norm(ltemp[idx]))
    iax.plot(np.real(itod),np.imag(itod),'-',c=c)
iax.set_xlabel('I')
iax.set_ylabel('Q')

iax = ax2[1,0]
im = iax.scatter(np.real(liq_norm),np.imag(liq_norm),c=ltemp,cmap='jet')
iax.set_xlabel('Pedestal I Noramlized')
iax.set_ylabel('Pedestal Q Normalized')

iax = ax2[1,1]
for idx,itod in enumerate(ltod_norm):
    c = im.cmap(im.norm(ltemp[idx]))
    a = 0.5
    iax.plot(np.real(itod),np.imag(itod),'-',alpha=a,c=c)
iax.set_xlabel('I Normalized')
iax.set_ylabel('Q Normalized')


for iax in ax2.flatten():
    iax.grid()
    cbar = fig2.colorbar(im, ax=iax)
    cbar.set_label('Temperature [K]')

for idx in range(2):
    ax2[1,idx].set_xlim(0,1)
    ax2[1,idx].set_ylim(-0.5,0.5)

print(lts)
print(ltemp)

fig2.tight_layout()
plt.show()



