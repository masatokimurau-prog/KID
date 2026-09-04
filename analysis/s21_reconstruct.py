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

    # df = pd.DataFrame(dres)
    # print(df)
    # df.to_csv(fn_.replace('.npz','.csv'),index=False)
    np.savez(fn_.replace('.npz','_res.npz'),**dres)


def Analysis(fn_):
    res = np.load(fn_)
    print(res.files)

    binped = np.linspace(-10.0,10.0,101,endpoint=True)
    binc = np.linspace(-2000.0,2000.0,101,endpoint=True)

    ##### pedestal
    fig,ax = plt.subplots(figsize=(16,9),ncols=2,nrows=2)
    ax = ax.flatten()
    h2D,_,_ = np.histogram2d(res['ch0_ped'],res['ch1_ped'],bins=[binped,binped])
    cl = ax[0].imshow(np.ma.masked_where(h2D.T==0, h2D.T),origin='lower',extent=[binped[0],binped[-1],binped[0],binped[-1]],aspect='auto',cmap='jet')
    fig.colorbar(cl,ax=ax[0])
    ax[0].set_xlabel('ch0 pedestal [mV]')
    ax[0].set_ylabel('ch1 pedestal [mV]')
    ax[0].grid()

    cl = ax[1].imshow(np.ma.masked_where(h2D==0, h2D),origin='lower',extent=[binped[0],binped[-1],binped[0],binped[-1]],aspect='auto',cmap='jet')
    fig.colorbar(cl,ax=ax[1])
    ax[1].set_xlabel('ch1 pedestal [mV]')
    ax[1].set_ylabel('ch0 pedestal [mV]')
    ax[1].grid()

    for ich in range(2):
        h,_ = np.histogram(res[f'ch{ich}_ped'],bins=binped)
        ax[2].hist(binped[:-1],binped[1:],weights=h,histtype='stepfilled',alpha=0.7,label=f'ch{ich}')
        h,_ = np.histogram(res[f'ch{ich}_integ200'],bins=binc)
        ax[3].hist(binc[:-1],binc[1:],weights=h,histtype='stepfilled',alpha=0.7,label=f'ch{ich}')
    ax[2].set_xlabel('pedestal [mV]')
    ax[3].set_xlabel('charge in [0,200] ns')
    for iax in ax[2:4]:
        iax.set_ylabel('counts')
        iax.grid()
        iax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    fn = sys.argv[1]
    if 'npz' in fn and 'res' not in fn:
        ProcessData(fn)
    elif 'res.npz' in fn:
        Analysis(fn)
