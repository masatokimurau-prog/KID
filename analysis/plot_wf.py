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

def single_file(filename_, pdf1_=None):
    data = np.load(filename_)
    for ikey in data.keys():
        print(ikey)
    print('number of points =',data['npts'])
    print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])

    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    tbin = (np.arange(npts) - npts*ref_position/100)/sample_rate
    nwf = data['ch1'].shape[0]

    print(ref_position)
    print(tbin)

    linteg = np.sum(data['ch0'][:,(int)(npts*ref_position/100):(int)(npts*ref_position/100)+100],axis=1)
    # linteg = np.sum(data['ch1'][:,(int)(npts*ref_position/100):(int)(npts*ref_position/100)+2500],axis=1)
    lped   = data['ch1'][:,:(int)(npts*ref_position/2/100)].mean(axis=1)
    binped = np.linspace(data['ch1'].min(),data['ch1'].max(),100)
    # binped = np.arange(-0.08,-0.04001,0.0005)
    # binped = np.arange(-0.11,-0.07001,0.0005)
    # binped = np.arange(-0.005,0.05001,0.0005)
    # binped = np.arange(0.08,0.12001,0.0005)
    if np.mean(linteg) < 0:
        binq   = np.linspace(linteg.min(),0,100)
    else:
        binq   = np.linspace(0,linteg.max(),100)

    print(len(data['ch0']), len(lped), len(linteg))

    h2d,_,_ = np.histogram2d(lped,linteg,bins=(binped,binq))
    hped,_ = np.histogram(lped,bins=binped)
    
    sig_th = 0.002
    ped_th = -0.065
    lmax = data['ch0'].max(axis=1)
    lmin = data['ch0'].min(axis=1)
    # sig_idx = np.where(((lmax>sig_th)&(lped<ped_th)))[0]
    # sig_idx = np.where(((lmax>sig_th)))[0]
    # sig_idx = np.where(((lmin<sig_th)))[0]
    sig_idx = np.arange(590,599)

    all_ave = data['ch0'][:,:].mean(axis=0)
    sig_ave = data['ch0'][sig_idx,:].mean(axis=0)
    
    pdf1 = PdfPages('pc1.pdf') if pdf1_ is None else pdf1_
    
    ncol = 3
    nrow = 3
    fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=False)
    ax = ax.flatten()
    ch0_min = data['ch0'].min()
    ch0_max = data['ch0'].max()
    for idx in range(ncol*nrow):
        ax[idx].plot(tbin*1e9,data[f'ch0'][idx],'-',label=f'Ch0',c='C0',alpha=0.7)
        ax_r = ax[idx].twinx()
        ax_r.plot(tbin*1e9,data[f'ch1'][idx],'-',label=f'Ch1',c='C1',alpha=0.7)
        ax[idx].grid()
        # ax[idx].legend()
        ax[idx].tick_params(colors='C0')
        ax_r.tick_params(colors='C1')
        # ax[idx].set_xlim(0,100)
        ax[idx].set_ylim(0,0.02)
        ax_r.set_ylim(-0.015,0.005)
        ax[idx].set_title(f'Event #{idx}')
    
    for ic in range(ncol):
        ax[ncol*(nrow-1)+ic].set_xlabel('Time [ns]')
    for ir in range(nrow):
        ax[ir*ncol].set_ylabel('Voltage [mV]')

    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')


    fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=False)
    ax = ax.flatten()
    ch0_min = data['ch0'].min()
    for idx,jdx in enumerate(sig_idx[:ncol*nrow]):
        ax[idx].plot(tbin*1e9,data[f'ch0'][jdx],'-',label=f'Ch0',c='C0',alpha=0.7)
        ax_r = ax[idx].twinx()
        ax_r.plot(tbin*1e9,data[f'ch1'][jdx],'-',label=f'Ch1',c='C1',alpha=0.7)
        ax[idx].grid()
        # ax[idx].legend()
        ax[idx].tick_params(colors='C0')
        ax_r.tick_params(colors='C1')
        ax[idx].set_ylim(ch0_min,ch0_min+0.02)
        ax_r.set_ylim(-0.01,0.01)
        ax[idx].set_title(f'Event #{jdx}')
    
    for ic in range(ncol):
        ax[ncol*(nrow-1)+ic].set_xlabel('Time [ns]')
    for ir in range(nrow):
        ax[ir*ncol].set_ylabel('Voltage [mV]')

    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    plt.show()

    # fig,ax = plt.subplots(figsize=(8,6))
    # rb = 10
    # for idx in range(5):
    #     ax.plot(tbin.reshape(-1,rb).mean(axis=1)*1e9, data[f'ch0'][2*idx].reshape(-1,rb).mean(axis=1)*1e3, '-',label=f'Event #{idx}',alpha=0.75)
    # ax.set_xlabel('Time [ns]')
    # ax.set_ylabel('Voltage [mV]')
    # ax.set_xlim(-200,1500)
    # ax.legend()

    # fig.tight_layout()
    # fig.savefig(pdf1,format='pdf')

    # fig,ax = plt.subplots(figsize=(16,9),nrows=2,sharex=True)
    # for idx in sig_idx:
    #     ax[0].plot(tbin*1e9,data['ch0'][idx],'-',c='C0',alpha=0.1)
    #     ax[1].plot(tbin*1e9,data['ch1'][idx],'-',c='C0',alpha=0.1)
    # for iax in ax:
    #     iax.set_xlim(-100,1000)
    #     iax.set_ylabel('voltage [mV]')
    #     iax.grid()
    # ax[1].set_xlabel('Time [ns]')
    # fig.tight_layout()
    # fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(tbin*1e9,sig_ave,'-',label='Above threshold')
    ax.plot(tbin*1e9,all_ave,'-',label='All triggered')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('voltage [mV]')
    ax.set_xlim(-100,1000)
    ax.grid()
    ax.legend()
    fig.savefig(pdf1,format='pdf')


    fig,ax = plt.subplots(figsize=(16,9))
    im = ax.pcolormesh(*np.meshgrid(binped,binq),np.ma.masked_where(h2d.T<=0,h2d.T))
    ax.set_xlabel('ch0 Pedestal [mV]')
    ax.set_ylabel('ch1 Integral [mV*bin]')
    ax.grid()
    fig.colorbar(im)    
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')


    '''
    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(np.arange(len(lped)),lped,'o-',c='C0')
    ax_r = ax.twinx()
    # ax_r.plot(np.arange(len(linteg)),linteg,'o-',c='C1')
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Pedestal [mV]')
    ax_r.set_ylabel('Integrated charge')
    ax.tick_params(colors='C0')
    ax_r.tick_params(colors='C1')
    ax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
    '''
    
    if pdf1_ is None:
        pdf1.close()

    return {'tbin':tbin,'sig_ave':sig_ave,'all_ave':all_ave,'binped':binped,'hped':hped,'lped':lped,'linteg':linteg,'lmax':lmax,'lmin':lmin}


def dir_files(dirname_):
    files = glob.glob(f'{dirname_}/*.npz')
    print(files)
    ghzs = []
    for ifile in files:
        ghz = re.findall('.*_([0-9]*\.[0-9]*)Hz\.npz',ifile)
        print(ifile, ghz[0])
        ghzs.append(float(ghz[0]))
        
    files = [val for _,val in sorted(zip(ghzs,files))]
    ghzs.sort()

    files = files[::10]
    ghzs = ghzs[::10]
    
    pdf11 = PdfPages('pc11.pdf')
    dtbin, dsig_ave, dall_ave, dhped, dlped, dlinteg, dlmax, dlmin = {}, {}, {}, {}, {}, {}, {}, {}

    for ifile,ighz in zip(files,ghzs):
        res = single_file(ifile,pdf11)
        dtbin[ighz] = res['tbin']
        dsig_ave[ighz] = res['sig_ave']
        dall_ave[ighz] = res['all_ave']
        dhped[ighz] = res['hped']
        dlped[ighz] = res['lped']
        dlinteg[ighz] = res['linteg']
        dlmin[ighz] = res['lmin']
        dlmax[ighz] = res['lmax']

    lped_all, linteg_all, lmax_all, lmin_all = [], [], [], []
    for ighz in ghzs:
        lped_all.extend(dlped[ighz])
        linteg_all.extend(dlinteg[ighz])
        lmax_all.extend(dlmax[ighz])
        lmin_all.extend(dlmin[ighz])
    lped_all = np.array(lped_all)
    linteg_all = np.array(linteg_all)
    lmax_all = np.array(lmax_all)
    lmin_all = np.array(lmin_all)

    binped = np.linspace(lped_all.min(),lped_all.max(),100)
    bininteg = np.linspace(linteg_all.min(),linteg_all.max(),100)
    binmax = np.linspace(lmax_all.min(),lmax_all.max(),100)
    binmin = np.linspace(lmin_all.min(),lmin_all.max(),100)

    h2d_integ,_,_ = np.histogram2d(lped_all,linteg_all,bins=(binped,bininteg))
    h2d_max,_,_ = np.histogram2d(lped_all,lmax_all,bins=(binped,binmax))
    h2d_min,_,_ = np.histogram2d(lped_all,lmin_all,bins=(binped,binmin))
    
        
    pdf12 = PdfPages('pc12.pdf')
    pdf13 = PdfPages('pc13.pdf')
    fig,ax = plt.subplots(figsize=(16,9),nrows=2,sharex=True)
    for idx,ighz in enumerate(ghzs):
        ax[0].plot(dtbin[ighz]*1e9,dall_ave[ighz],'-',c=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} Hz')
        ax[1].plot(dtbin[ighz]*1e9,dsig_ave[ighz],'-',c=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} Hz')
    
    leg_opt = {'columnspacing':1,'fontsize':'small','handlelength':1,'ncols':3}
    ax[0].legend(**leg_opt,title='All triggered')
    ax[1].legend(**leg_opt,title='Above threshold')

    ax[1].set_xlabel('Time [ns]')
    for iax in ax:
        iax.set_ylabel('voltage [mV]')
        iax.grid()
    fig.savefig(pdf12,format='pdf')

    for iax in ax: iax.set_xlim(-100,1000)
    fig.savefig(pdf12,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9))
    for idx,(ighz,dhped) in enumerate(dhped.items()):
        ax.hist(res['binped'][:-1],res['binped'],weights=dhped,histtype='step',color=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} GHz')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Pedestal [mV]')
    fig.savefig(pdf12,format='pdf')
    '''
    fig,ax = plt.subplots(figsize=(16,9))
    for idx,ighz in enumerate(ghzs):
        ax.plot(dlped[ighz],dlinteg[ighz],'.',color=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} Hz',alpha=0.2)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Pedestal [mV]')
    ax.set_ylabel('Integraled charge [mV*ns]')
    fig.savefig(pdf12,format='pdf')
    
    fig,ax = plt.subplots(figsize=(16,9))
    for idx,ighz in enumerate(ghzs):
        ax.plot(dlped[ighz],dlmax[ighz],'.',color=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} Hz',alpha=0.2)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Pedestal [mV]')
    ax.set_ylabel('Maximum [mV]')
    fig.savefig(pdf12,format='pdf')
    
    fig,ax = plt.subplots(figsize=(16,9))
    for idx,ighz in enumerate(ghzs):
        ax.plot(dlped[ighz],dlmin[ighz],'.',color=mpl.cm.jet(idx/len(ghzs)),label=f'{ighz} Hz',alpha=0.2)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Pedestal [mV]')
    ax.set_ylabel('Minimum [mV]')
    fig.savefig(pdf12,format='pdf')
    '''
    fig,ax = plt.subplots(figsize=(8,14),nrows=3,sharex=True)
    im = []
    iim = ax[0].pcolormesh(*np.meshgrid(binped,bininteg),np.ma.masked_where(h2d_integ.T<=0,h2d_integ),norm=mpl.colors.LogNorm())
    im.append(iim)
    iim = ax[1].pcolormesh(*np.meshgrid(binped,binmax),np.ma.masked_where(h2d_max.T<=0,h2d_max),norm=mpl.colors.LogNorm())
    im.append(iim)
    iim = ax[2].pcolormesh(*np.meshgrid(binped,binmin),np.ma.masked_where(h2d_min.T<=0,h2d_min),norm=mpl.colors.LogNorm())
    im.append(iim)

    ax[0].set_ylabel('Integral [mV*ns]')
    ax[1].set_ylabel('Maximum [mV]')
    ax[2].set_ylabel('Minimum [mV]')

    for iax,iim in zip(ax,im):
        fig.colorbar(iim,ax=iax)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf13,format='pdf')

    pdf11.close()
    pdf12.close()
    pdf13.close()
        

    
    
if __name__ == '__main__':
    if 'npz' in sys.argv[1]:
        single_file(sys.argv[1])
    else:
        dir_files(sys.argv[1])
    

