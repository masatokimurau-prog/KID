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
from scipy.optimize import curve_fit  
import quadpy

import ds_style

def funcdex(X_, tau_, rise_):
    tmax = np.log(rise_/tau_)/(1/tau_-1/rise_)
    return np.where(X_>=0,(np.exp(-X_/tau_)-np.exp(-X_/rise_))/(np.exp(-tmax/tau_)-np.exp(-tmax/rise_)),0)

def funcfit(X_, t0_, k_, tau_, rise_, ped_):
    return k_*funcdex(X_-t0_,tau_,rise_) + ped_

def single_file(filename_, pdf1_=None):
    data = np.load(filename_,allow_pickle=True)
    for ikey in data.keys():
        print(ikey)
    print('number of points =',data['npts'])
    print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])

    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    tbin = (np.arange(npts) - npts*ref_position/100)/sample_rate
    nwf = data['ch1'].shape[0]
    print(type(data['deltat']))
    deltat = (np.array(data['deltat'],dtype='timedelta64[ms]')/np.timedelta64(1,'ms')).astype(int)

    print(ref_position)
    print(tbin)

    linteg = np.abs(np.sum(data['ch0'][:,(int)(npts*ref_position/100):(int)(npts*ref_position/100)+100],axis=1))
    # linteg = np.sum(data['ch1'][:,(int)(npts*ref_position/100):(int)(npts*ref_position/100)+2500],axis=1)
    lped   = data['ch1'][:,:(int)(npts*ref_position/2/100)].mean(axis=1)
    lpedrms = data['ch1'][:,:(int)(npts*ref_position/2/100)].std(axis=1)
    binped = np.linspace(data['ch1'].min(),data['ch1'].max(),100)
    binq   = np.linspace(0,linteg.max(),100)

    print(len(data['ch0']), len(lped), len(linteg))

    h2d,_,_ = np.histogram2d(lped,linteg,bins=(binped,binq))
    hped,_ = np.histogram(lped,bins=binped)
    
    sig_th = -0.001
    # ped_th = -0.065
    lmax = data['ch0'].max(axis=1)
    lmin = data['ch0'].min(axis=1)
    # sig_idx = np.where(((lmax>sig_th)&(lped<ped_th)))[0]
    sig_idx = np.where(((lmin<sig_th)))[0]
    # sig_idx = np.where(((lmin<sig_th)))[0]

    all_ave = data['ch0'][:,:].mean(axis=0)
    sig_ave = data['ch0'][sig_idx,:].mean(axis=0)

    lpopt = []
    for idx in sig_idx:
        # print(idx)
        try:
            popt0,pcov0 = curve_fit(funcfit, tbin*1e9, data['ch0'][idx], p0=[0,data['ch0'][idx].min(),500,20,0])
        except RuntimeError:
            popt0 = [0,0,0,0,0]
            pcov0 = [0 for _ in range(5) for _ in range(5)]
        # print(popt0)
        try:
            popt1,pcov1 = curve_fit(funcfit, tbin*1e9, data['ch1'][idx], p0=[0,data['ch1'][idx].max(),500,20,lped[idx]], maxfev=10000, bounds=([-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,1000,200,np.inf]))
        except RuntimeError:
            popt1 = [0,0,0,0,0]
            pcov1 = [0 for _ in range(5) for _ in range(5)]
        # print(popt1)
        lpopt.append([popt0,popt1])

    lpopt = np.array(lpopt)
    df_fit = pd.DataFrame(lpopt.reshape(lpopt.shape[0],-1), columns = [f'ch{ich}_{itag}' for ich in range(2) for itag in ['t0','k','tau','rise','ped']])

    for ich in range(2):
        df_fit[f'ch{ich}_t0'] = df_fit[f'ch{ich}_t0']*1e-3
        df_fit[f'ch{ich}_k'] = df_fit[f'ch{ich}_k']*1e3
        df_fit[f'ch{ich}_tau'] = df_fit[f'ch{ich}_tau']*1e-3
        df_fit[f'ch{ich}_rise'] = df_fit[f'ch{ich}_rise']*1e-3
        df_fit[f'ch{ich}_ped'] = df_fit[f'ch{ich}_ped']*1e3
    df_fit['absk'] = np.hypot(df_fit['ch0_k'].values,df_fit['ch1_k'].values)
    df_fit['dt_ms'] = deltat[sig_idx]
    print(df_fit)

    # print(np.exp(-0.5/df_fit['ch0_tau']),np.exp(-0.5/df_fit['ch0_rise']),np.exp(-0.5/df_fit['ch0_tau'])-np.exp(-0.5/df_fit['ch0_rise']),df_fit['ch0_k'][0]*(np.exp(-0.5/df_fit['ch0_tau'])-np.exp(-0.5/df_fit['ch0_rise'])),funcfit(500,*lpopt[0][0]))
    
    ##### Make canvas
    pdf1 = PdfPages('pc1.pdf') if pdf1_ is None else pdf1_
    
    ncol = 4
    nrow = 4
    fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=False)
    ax = ax.flatten()
    ch0_min = data['ch0'].min()
    ch1_max = max(-data['ch1'].min(),data['ch1'].max())
    for idx in range(ncol*nrow):
        ax[idx].plot(tbin*1e9,data[f'ch0'][idx],'-',label=f'Ch0',c='C0',alpha=0.7)
        ax_r = ax[idx].twinx()
        ax_r.plot(tbin*1e9,data[f'ch1'][idx],'-',label=f'Ch1',c='C1',alpha=0.7)
        ax[idx].grid()
        # ax[idx].legend()
        ax[idx].tick_params(colors='C0')
        ax_r.tick_params(colors='C1')
        ax[idx].set_ylim(ch0_min,0.001)
        ax_r.set_ylim(-ch1_max,ch1_max)
    
    for ic in range(ncol):
        ax[ncol*(nrow-1)+ic].set_xlabel('Time [ns]')
    for ir in range(nrow):
        ax[ir*ncol].set_ylabel('Voltage [V]')

    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    
    fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=False)
    ax = ax.flatten()
    for idx in range(ncol*nrow):
        ax[idx].plot(tbin*1e9,data[f'ch0'][sig_idx[idx]],'-',label=f'Ch0',c='C0',alpha=0.7)
        ax[idx].plot(tbin*1e9,funcfit(tbin*1e9,*lpopt[idx][0]),'-',c='k')
        ax_r = ax[idx].twinx()
        ax_r.plot(tbin*1e9,data[f'ch1'][sig_idx[idx]],'-',label=f'Ch1',c='C1',alpha=0.7)
        ax_r.plot(tbin*1e9,funcfit(tbin*1e9,*lpopt[idx][1]),'-',c='k')
        ax[idx].grid()
        # ax[idx].legend()
        ax[idx].tick_params(colors='C0')
        ax_r.tick_params(colors='C1')
        ax[idx].set_ylim(ch0_min,0.001)
        ax_r.set_ylim(-ch1_max,ch1_max)
    
    for ic in range(ncol):
        ax[ncol*(nrow-1)+ic].set_xlabel('Time [ns]')
    for ir in range(nrow):
        ax[ir*ncol].set_ylabel('Voltage [V]')

    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9),nrows=2,sharex=True)
    for idx in sig_idx:
        ax[0].plot(tbin*1e9,data['ch0'][idx],'-',c='C0',alpha=0.1)
        ax[1].plot(tbin*1e9,data['ch1'][idx],'-',c='C0',alpha=0.1)
    for iax in ax:
        iax.set_xlim(-100,1000)
        iax.set_ylabel('voltage [V]')
        iax.grid()
    ax[1].set_xlabel('Time [ns]')
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(tbin*1e9,sig_ave,'-',label='Above threshold')
    ax.plot(tbin*1e9,all_ave,'-',label='All triggered')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('voltage [V]')
    ax.set_xlim(-100,1000)
    ax.grid()
    ax.legend()
    fig.savefig(pdf1,format='pdf')


    '''
    fig,ax = plt.subplots(figsize=(16,9),ncols=2,nrows=2)
    ax = ax.flatten()
    im = ax[0].pcolormesh(*np.meshgrid(binped,binq),np.ma.masked_where(h2d.T<=0,h2d.T))
    ax[0].set_xlabel('ch1 Pedestal [mV]')
    ax[0].set_ylabel('ch0 Integral [mV*bin]')
    ax[0].grid()
    fig.colorbar(im,ax=ax[0])    
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
    '''

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

    fig,ax = plt.subplots(figsize=(16,9),ncols=2,nrows=2,sharex=True)
    ax = ax.flatten()
    H = ax[0].hist2d(df_fit.ch1_ped,df_fit.ch0_k,bins=[np.linspace(-5,15,50),np.linspace(-20,0,50)],cmap=mpl.cm.jet)
    ax[0].set_ylabel('Ch0 Peak [mV]')
    H = ax[1].hist2d(df_fit.ch1_ped,df_fit.ch0_rise,bins=[np.linspace(-5,15,50),np.linspace(0,0.2,50)],cmap=mpl.cm.jet)
    ax[1].set_ylabel('Ch0 Rise Time [us]')
    H = ax[2].hist2d(df_fit.ch1_ped,df_fit.ch0_tau,bins=[np.linspace(-5,15,50),np.linspace(0,1,50)],cmap=mpl.cm.jet)
    ax[2].set_ylabel('Ch0 Tau [us]')
    H = ax[3].hist2d(df_fit.ch1_ped,df_fit.ch0_t0,bins=[np.linspace(-5,15,50),np.linspace(-0.05,0.05,50)],cmap=mpl.cm.jet)
    ax[3].set_ylabel('Ch0 t0 [us]')
    for iax in ax[2:4]:
        iax.set_xlabel('Ch1 Pedestal [mV]')
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
        
    
    fig,ax = plt.subplots(figsize=(16,9),ncols=3,nrows=3)
    H = ax[0,0].hist2d(df_fit.ch0_k,df_fit.ch0_rise,bins=[np.linspace(-10,0,50),np.linspace(0,0.2,50)],cmap=mpl.cm.jet)
    ax[0,0].set_xlabel('Ch0 Peak [mV]')
    ax[0,0].set_ylabel('Ch0 Rise Time [us]')
    fig.colorbar(H[3],ax=ax[0,0])
    H = ax[0,1].hist2d(df_fit.ch0_k,df_fit.ch1_k,bins=[np.linspace(-10,0,50),np.linspace(-10,10,50)],cmap=mpl.cm.jet)
    ax[0,1].set_xlabel('Ch0 Peak [mV]')
    ax[0,1].set_ylabel('Ch1 Peak [mV]')
    fig.colorbar(H[3],ax=ax[0,1])
    H = ax[0,2].hist2d(df_fit.ch0_rise,df_fit.ch1_rise,bins=[np.linspace(0,0.2,50),np.linspace(0,0.2,50)],cmap=mpl.cm.jet)
    ax[0,2].set_xlabel('Ch0 Rise Time [us]')
    ax[0,2].set_ylabel('Ch1 Rise Time [us]')
    fig.colorbar(H[3],ax=ax[0,2])
    ax[1,0].hist(df_fit.ch0_k.abs(),bins=np.linspace(0,20,50),label='Ch0',alpha=0.6,histtype='step')
    ax[1,0].hist(df_fit.ch1_k.abs(),bins=np.linspace(0,20,50),label='Ch1',alpha=0.6,histtype='step')
    ax[1,0].hist(df_fit.absk,bins=np.linspace(0,20,50),label='Abs',alpha=0.6,histtype='step')
    ax[1,0].set_xlabel('|Peak| [mV]')
    ax[1,0].legend()
    ax[1,1].hist(df_fit.ch0_rise,bins=np.linspace(0,0.2,50),label='Ch0',alpha=0.6,histtype='step')
    ax[1,1].hist(df_fit.ch1_rise,bins=np.linspace(0,0.2,50),label='Ch1',alpha=0.6,histtype='step')
    ax[1,1].set_xlabel('Rise Time [us]')
    ax[1,1].legend()
    ax[1,2].hist(df_fit.ch0_tau,bins=np.linspace(0,2,50),label='Ch0',alpha=0.6,histtype='step')
    ax[1,2].hist(df_fit.ch1_tau,bins=np.linspace(0,2,50),label='Ch1',alpha=0.6,histtype='step')
    ax[1,2].set_xlabel('Tau [us]')
    ax[1,2].legend()
    H = ax[2,0].hist2d(df_fit.ch1_ped,df_fit.ch0_k,bins=[np.linspace(-5,15,50),np.linspace(-20,0,50)],cmap=mpl.cm.jet)
    ax[2,0].set_xlabel('Ch1 Pedestal [mV]')
    ax[2,0].set_ylabel('Ch0 Peak [mV]')
    fig.colorbar(H[3],ax=ax[2,0])
    H = ax[2,1].hist2d(df_fit.ch1_ped,df_fit.ch0_k,bins=[np.linspace(-5,15,50),np.linspace(-20,10,50)],cmap=mpl.cm.jet)
    ax[2,1].set_xlabel('Ch1 Pedestal [mV]')
    ax[2,1].set_ylabel('Ch1 Peak [mV]')
    fig.colorbar(H[3],ax=ax[2,1])
    H = ax[2,2].hist2d(df_fit.ch0_tau,df_fit.ch1_tau,bins=[np.linspace(0,1,50),np.linspace(0,1,50)],cmap=mpl.cm.jet)
    ax[2,2].set_xlabel('Ch0 Tau [us]')
    ax[2,2].set_ylabel('Ch1 Tau [us]')
    fig.colorbar(H[3],ax=ax[2,2])
    # H = ax[2,2].hist2d(df_fit.ch1_ped,df_fit.ch0_k,bins=[np.linspace(-10,10,50),np.linspace(-10,0,50)],cmap=mpl.cm.jet)
    # ax[2,2].set_xlabel('Ch1 Pedestal [mV]')
    # ax[2,2].set_ylabel('Ch0 Peak [mV]')
    # fig.colorbar(H[3],ax=ax[2,2])
    
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
    
    
    if pdf1_ is None:
        pdf1.close()

    
    resname = filename_.replace('.npz','_fitres.csv')
    df_fit.to_csv(resname)

        
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
    iim = ax[0].pcolormesh(*np.meshgrid(binped,bininteg),np.ma.masked_where(h2d_integ.T<=0,h2d_integ.T),norm=mpl.colors.LogNorm())
    im.append(iim)
    iim = ax[1].pcolormesh(*np.meshgrid(binped,binmax),np.ma.masked_where(h2d_max.T<=0,h2d_max.T),norm=mpl.colors.LogNorm())
    im.append(iim)
    iim = ax[2].pcolormesh(*np.meshgrid(binped,binmin),np.ma.masked_where(h2d_min.T<=0,h2d_min.T),norm=mpl.colors.LogNorm())
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
    

