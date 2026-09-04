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

MYVERBOSE_ = False

def funcdex(X_, tau_, rise_):
    tmax = np.log(rise_/tau_)/(1/tau_-1/rise_)
    return np.where(X_>=0,(np.exp(-X_/tau_)-np.exp(-X_/rise_))/np.abs(np.exp(-tmax/tau_)-np.exp(-tmax/rise_)),0)

def funcfit(X_, t0_, k_, tau_, rise_, ped_):
    return k_*funcdex(X_-t0_,tau_,rise_) + ped_

def plot_raw_wf(tbin_,data_,lpopt_,sig_idx_,freq_,pdf0_):
    ncol = 4
    nrow = 4 
    ch0_range = (data_['ch0'].min()*1e3,data_['ch0'].max()*1e3)
    ch1_range = (data_['ch1'].min()*1e3,data_['ch1'].max()*1e3)
    rep_t = np.linspace(0,len(tbin_),20,False).astype(int)[2:-2]
    
    fig,ax = plt.subplots(ncols=ncol,nrows=nrow,sharex=True,sharey=True)
    fig2,ax2 = plt.subplots(ncols=ncol,nrows=nrow,sharex=True,sharey=True)
    ax = ax.flatten()
    ax2 = ax2.flatten()
    for idx in range(min(ncol*nrow,len(sig_idx_))):
        ax[idx].plot(tbin_*1e9,data_[f'ch0'][sig_idx_[idx]]*1e3,'-',label=f'Ch0',c='C0',alpha=0.7)
        ax[idx].plot(tbin_*1e9,funcfit(tbin_*1e9,*lpopt_[sig_idx_[idx]][0])*1e3,'-',c='k')
        ax_r = ax[idx].twinx()
        ax_r.plot(tbin_*1e9,data_[f'ch1'][sig_idx_[idx]]*1e3,'-',label=f'Ch1',c='C1',alpha=0.7)
        ax_r.plot(tbin_*1e9,funcfit(tbin_*1e9,*lpopt_[sig_idx_[idx]][1])*1e3,'-',c='k')
        for jc,jdx in enumerate(rep_t):
            ax[idx].plot(tbin_[jdx]*1e9,data_[f'ch0'][sig_idx_[idx]][jdx]*1e3,'o',ms=5,c=mpl.cm.jet(jc/len(rep_t)))
            ax_r.plot(tbin_[jdx]*1e9,data_[f'ch1'][sig_idx_[idx]][jdx]*1e3,'o',ms=5,c=mpl.cm.jet(jc/len(rep_t)))
        ax[idx].grid()
        # ax[idx].legend()
        ax[idx].tick_params(axis='y',colors='C0')
        ax_r.tick_params(colors='C1')
        ax[idx].set_ylim(ch0_range)
        ax_r.set_ylim(ch1_range)
        if idx%nrow!=ncol-1:
            ax_r.tick_params(labelright=False)

        ax2[idx].plot(data_[f'ch0'][sig_idx_[idx]]*1e3,data_[f'ch1'][sig_idx_[idx]]*1e3,'-',c='k')
        for jc,jdx in enumerate(rep_t):
            ax2[idx].plot(data_[f'ch0'][sig_idx_[idx]][jdx]*1e3,data_[f'ch1'][sig_idx_[idx]][jdx]*1e3,'o',ms=5,c=mpl.cm.jet(jc/len(rep_t))) 
        ax2[idx].grid()
        ax2[idx].set_xlim(ch0_range)
        ax2[idx].set_ylim(ch1_range)
    
    for ic in range(ncol):
        ax[ncol*(nrow-1)+ic].set_xlabel('Time [us]')
        ax2[ncol*(nrow-1)+ic].set_xlabel('ch0 [mV]')
    for ir in range(nrow):
        ax[ir*ncol].set_ylabel('Voltage [mV]')
        ax2[ir*ncol].set_ylabel('ch1 [mV]')
    if freq_>0:
        ax[0].set_title(f'{freq_} GHz')
        ax2[0].set_title(f'{freq_} GHz')

    fig.tight_layout()
    fig.savefig(pdf0_,format='pdf')
    fig2.tight_layout()
    fig2.savefig(pdf0_,format='pdf')

    
    

def single_file(filename_, rebin_fac_, pdf0_=None, freq_=0):
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
    deltat = (np.array(data['deltat'],dtype='timedelta64[ms]')/np.timedelta64(1,'ms')).astype(int)

    print(sample_rate,npts,ref_position)
    print(tbin)

    
    ##### rebin
    if npts%rebin_fac_>0:
        print(f'Error:Tried to rebin{rebin_fac_} for {npts} points')
        return
    data_rebin = {}
    for ich in range(2):
        data_rebin[f'ch{ich}'] = data[f'ch{ich}'].reshape(nwf,-1,rebin_fac_).mean(axis=2)
        print(data[f'ch{ich}'].shape, data_rebin[f'ch{ich}'].shape)
    tbin_rebin = tbin.reshape(-1,rebin_fac_).mean(axis=1)
    data_rebin['npts'] = npts/rebin_fac_
    npts_rebin = data_rebin['npts']

    
    llinteg, llped, llpedrms, =  [], [], []
    for ich in range(2):
        linteg = np.abs(np.sum(data_rebin[f'ch{ich}'][:,(int)(npts_rebin*ref_position/100):(int)(npts_rebin*ref_position/100+500/rebin_fac_)],axis=1))
        lped   = data_rebin[f'ch{ich}'][:,:(int)(npts_rebin*ref_position/2/100)].mean(axis=1)
        lpedrms = data_rebin[f'ch{ich}'][:,:(int)(npts_rebin*ref_position/2/100)].std(axis=1)

        llinteg.append(linteg)
        llped.append(lped)
        llpedrms.append(lpedrms)

    llinteg = np.array(llinteg)
    llped = np.array(llped)
    llpedrms = np.array(llpedrms)

    
    binped = np.linspace(llped.min(),llped.max(),100)
    binq   = np.linspace(0,max(-llinteg.min(),llinteg.max()),100)
    bink   = np.linspace(-5,5,50)
    binrise = np.linspace(0,0.2,50)
    bintau = np.linspace(0,1.1,100)

    print(len(data['ch0']), len(lped), len(linteg))

    h2d,_,_ = np.histogram2d(lped,linteg,bins=(binped,binq))
    hped,_ = np.histogram(lped,bins=binped)
    
    sig_th = -0.001
    # ped_th = -0.065
    # lmax = data['ch0'].max(axis=1)
    # lmin = data['ch0'].min(axis=1)
    # sig_idx = np.where(((lmax>sig_th)&(lped<ped_th)))[0]
    # sig_idx = np.where(((lmin<sig_th)))[0]
    # sig_idx = np.where(((lmin<sig_th)))[0]

    all_ave_0 = data['ch0'][:,:].mean(axis=0)
    all_ave_1 = data['ch1'][:,:].mean(axis=0)

    lllped = np.repeat(llped[:,:,np.newaxis], npts, axis=2)
    all_ave_corr0 = (data['ch0']-lllped[0])[:,:].mean(axis=0)
    all_ave_corr1 = (data['ch1']-lllped[1])[:,:].mean(axis=0)

    lpopt = []
    linfo = []
    for idx in range(nwf):
        ped0 = llped[0][idx]
        pol = np.sign(data_rebin['ch0'][idx][(int)(500/rebin_fac_+npts_rebin*ref_position/100)] - ped0)
        k0 = data_rebin['ch0'][idx].max()-ped0 if pol>0 else data_rebin['ch0'][idx].min()-ped0
        p0 = [0, k0, 500, 50, ped0]
        bounds = ([0, 0.1*k0, 0, 0, -np.inf],[500, 5*k0, 2000, 500, np.inf]) if pol>0 else ([0, 5*k0, 0, 0, -np.inf],[500, 0.1*k0, 2000, 500, np.inf])
        if MYVERBOSE_: print('ch0',pol,p0, bounds)
        try:
            popt0,pcov0,info0,_,_ = curve_fit(funcfit, tbin_rebin*1e9, data_rebin['ch0'][idx], sigma=llpedrms[0][idx], p0=p0, bounds=bounds, maxfev=10000, full_output=True, method='trf')
        except RuntimeError:
            popt0 = [0,0,0,0,0]
            pcov0 = [0 for _ in range(5) for _ in range(5)]
        if MYVERBOSE_: print(popt0)
            
        ped0 = llped[1][idx]
        pol = np.sign(data_rebin['ch1'][idx][(int)(500/rebin_fac_+npts_rebin*ref_position/100)] - ped0)
        k0 = data_rebin['ch1'][idx].max()-ped0 if pol>0 else data_rebin['ch1'][idx].min()-ped0
        p0 = [0, k0, 500, 50, ped0]
        bounds = ([0, 0.1*k0, 0, 0, -np.inf],[500, 5*k0, 2000, 500, np.inf]) if pol>0 else ([0, 5*k0, 0, 0, -np.inf],[500, 0.1*k0, 2000, 500, np.inf])
        if MYVERBOSE_: print('ch1',pol,p0, bounds)
        try:
            popt1,pcov1,info1,_,_ = curve_fit(funcfit, tbin_rebin*1e9, data_rebin['ch1'][idx], sigma=llpedrms[1][idx], p0=p0, bounds=bounds, maxfev=10000, full_output=True, method='trf')
            # popt1,pcov1 = curve_fit(funcfit, tbin*1e9, data_rebin['ch1'][idx], p0=[0,data_rebin['ch1'][idx].min(),500,20,lped[idx]], maxfev=10000, bounds=([-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,1000,200,np.inf]))
        except RuntimeError:
            popt1 = [0,0,0,0,0]
            pcov1 = [0 for _ in range(5) for _ in range(5)]
        if MYVERBOSE_: print(popt1)
        lpopt.append([popt0,popt1])
        linfo.append([info0,info1])

    lfitpar = ['t0','k','tau','rise','ped']
    lpopt = np.array(lpopt)
    print(lpopt.shape)
    df_fit = pd.DataFrame(lpopt.reshape(lpopt.shape[0],-1), columns = [f'ch{ich}_{itag}' for ich in range(2) for itag in lfitpar])

    print(linfo[0][0].keys())
    for ich in range(2):
        df_fit[f'ch{ich}_t0'] = df_fit[f'ch{ich}_t0']*1e-3
        df_fit[f'ch{ich}_k'] = df_fit[f'ch{ich}_k']*1e3
        df_fit[f'ch{ich}_tau'] = df_fit[f'ch{ich}_tau']*1e-3
        df_fit[f'ch{ich}_rise'] = df_fit[f'ch{ich}_rise']*1e-3
        df_fit[f'ch{ich}_ped'] = df_fit[f'ch{ich}_ped']*1e3

        df_fit[f'ch{ich}_chi2'] = [np.sum(x[ich]['fvec']**2) for x in linfo] / (npts_rebin-len(popt0))
        df_fit[f'ch{ich}_pedrms'] = llpedrms[ich]
        
        # df_fit[f'ch{ich}_R2'] = 
        
    df_fit['absk'] = np.hypot(df_fit['ch0_k'].values,df_fit['ch1_k'].values)
    df_fit['dt_ms'] = deltat
    print(df_fit)

    # print(np.exp(-0.5/df_fit['ch0_tau']),np.exp(-0.5/df_fit['ch0_rise']),np.exp(-0.5/df_fit['ch0_tau'])-np.exp(-0.5/df_fit['ch0_rise']),df_fit['ch0_k'][0]*(np.exp(-0.5/df_fit['ch0_tau'])-np.exp(-0.5/df_fit['ch0_rise'])),funcfit(500,*lpopt[0][0]))
    
    ##### Make canvas
    pdf0 = PdfPages('pc0.pdf') if pdf0_ is None else pdf0_
    pdf1 = PdfPages('pc1.pdf')

    plot_raw_wf(tbin, data, lpopt, np.arange(16,32), freq_, pdf0)
    plot_raw_wf(tbin_rebin, data_rebin, lpopt, np.arange(16,32), freq_, pdf0)

    # df_cut = df_fit.query(f'ch0_rise>0.1 or ch1_rise>0.1')
    # print(df_cut)
    # plot_raw_wf(tbin, data, lpopt, df_cut.index, freq_, pdf0)
    

    print(df_fit.columns)
    fig,ax = plt.subplots(nrows=2,ncols=len(popt0)+1,sharex=False)
    for idx,icol in enumerate(lfitpar):
        bins = np.linspace(min(df_fit[f'ch0_{icol}'].min(),df_fit[f'ch1_{icol}'].min()),max(df_fit[f'ch0_{icol}'].max(),df_fit[f'ch1_{icol}'].max()),50)
        for ich in range(2):
            ax[ich,idx].hist(df_fit[f'ch{ich}_{icol}'],bins=bins,color=f'C{ich}')
        ax[-1,idx].set_xlabel(icol)
    bincost = np.linspace(0,5,50)
    for ich in range(2):
        ax[ich,-1].hist(df_fit[f'ch{ich}_chi2'],bins=bincost,color=f'C{ich}')
    ax[-1,-1].set_xlabel('reduced chi2')
    for iax in ax.flatten():
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
    
    
    fig,ax = plt.subplots(nrows=2,sharex=True)
    for idx in np.arange(min(50,nwf)):
        ax[0].plot(tbin*1e9,data['ch0'][idx],'-',c='C0',alpha=0.1)
        ax[1].plot(tbin*1e9,data['ch1'][idx],'-',c='C1',alpha=0.1)
    for iax in ax:
        iax.set_xlim(-100,1000)
        iax.set_ylabel('voltage [V]')
        iax.grid()
    ax[1].set_xlabel('Time [ns]')
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')
    
    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(tbin*1e9,all_ave_0,'-',label='All triggered (ch0)')
    ax.plot(tbin*1e9,all_ave_1,'-',label='All triggered (ch1)')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('voltage [V]')
    ax.set_xlim(-100,1000)
    ax.grid()
    ax.legend()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9))
    ax.plot(tbin*1e9,all_ave_corr0,'-',label='All triggered (ch0)')
    ax.plot(tbin*1e9,all_ave_corr1,'-',label='All triggered (ch1)')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('voltage [V]')
    ax.set_xlim(-100,1000)
    ax.grid()
    ax.legend()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(ncols=2,nrows=2)
    ax = ax.flatten()
    for ich in range(2):
        H = ax[ich].hist2d(df_fit[f'ch{ich}_k'],df_fit[f'ch{ich}_rise'],bins=[bink,binrise],cmap=mpl.cm.jet)
        fig.colorbar(H[3],ax=ax[ich])
        ax[ich].set_xlabel(f'Ch{ich} k [mV]')
        ax[ich].set_ylabel(f'Ch{ich} r [us]')
    H = ax[2].hist2d(df_fit['ch0_k'],df_fit['ch1_k'],bins=[bink,bink],cmap=mpl.cm.jet)
    fig.colorbar(H[3],ax=ax[2])
    ax[2].set_xlabel('ch0 k [mV]')
    ax[2].set_ylabel('ch1 k [mV]')
    H = ax[3].hist2d(df_fit['ch0_tau'],df_fit['ch1_tau'],bins=[bintau,bintau],cmap=mpl.cm.jet)
    fig.colorbar(H[3],ax=ax[3])
    ax[3].set_xlabel('ch0 tau [us]')
    ax[3].set_ylabel('ch1 tau [us]')
    
    fig.tight_layout()    
    fig.savefig(pdf1,format='pdf')
    
    
    fig,ax = plt.subplots(ncols=3,nrows=3)
    H = ax[0,0].hist2d(df_fit.ch0_k,df_fit.ch0_rise,bins=[np.linspace(-10,0,50),np.linspace(0,0.2,50)],cmap=mpl.cm.jet)
    ax[0,0].set_xlabel('Ch0 Peak [mV]')
    ax[0,0].set_ylabel('Ch0 Rise Time [us]')
    fig.colorbar(H[3],ax=ax[0,0])
    H = ax[0,1].hist2d(df_fit.ch0_k,df_fit.ch1_k,bins=[np.linspace(-10,0,50),np.linspace(-10,10,50)],cmap=mpl.cm.jet)
    ax[0,1].set_xlabel('Ch0 Peak [mV]')
    ax[0,1].set_ylabel('Ch1 Peak [mV]')
    fig.colorbar(H[3],ax=ax[0,1])
    H = ax[0,2].hist2d(df_fit.ch0_rise,df_fit.ch1_rise,bins=[binrise,binrise],cmap=mpl.cm.jet)
    ax[0,2].set_xlabel('Ch0 Rise Time [us]')
    ax[0,2].set_ylabel('Ch1 Rise Time [us]')
    fig.colorbar(H[3],ax=ax[0,2])
    ax[1,0].hist(df_fit.ch0_k.abs(),bins=np.linspace(0,20,50),label='Ch0',alpha=0.6,histtype='step')
    ax[1,0].hist(df_fit.ch1_k.abs(),bins=np.linspace(0,20,50),label='Ch1',alpha=0.6,histtype='step')
    ax[1,0].hist(df_fit.absk,bins=np.linspace(0,20,50),label='Abs',alpha=0.6,histtype='step')
    ax[1,0].set_xlabel('|Peak| [mV]')
    ax[1,0].legend()
    ax[1,1].hist(df_fit.ch0_rise,bins=binrise,label='Ch0',alpha=0.6,histtype='step')
    ax[1,1].hist(df_fit.ch1_rise,bins=binrise,label='Ch1',alpha=0.6,histtype='step')
    ax[1,1].set_xlabel('Rise Time [us]')
    ax[1,1].legend()
    ax[1,2].hist(df_fit.ch0_tau,bins=bintau,label='Ch0',alpha=0.6,histtype='step')
    ax[1,2].hist(df_fit.ch1_tau,bins=bintau,label='Ch1',alpha=0.6,histtype='step')
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
    
    
    if pdf0_ is None:
        pdf0.close()
    pdf1.close()
    
    resname = filename_.replace('.npz','_fitres.csv')
    df_fit.to_csv(resname)

        
    return {'tbin':tbin,'all_ave_0':all_ave_0,'all_ave_1':all_ave_1,'all_ave_corr0':all_ave_corr0,'all_ave_corr1':all_ave_corr1}


def freq_scan_files(rebin_fac_):
    dfreq = {
        "0508_1420": 5.561,
        "0508_1422": 5.526,
        "0508_1424": 5.516,
        "0508_1425": 5.506,
        "0508_1426": 5.501,
        "0508_1427": 5.496,
        "0508_1428": 5.491,
        "0508_1429": 5.486,
        "0508_1430": 5.481,
        "0508_1431": 5.476,
        "0508_1432": 5.471,
        "0508_1433": 5.470,
        "0508_1434": 5.469,
        "0508_1435": 5.468,
        "0508_1436": 5.467,
        "0508_1437": 5.466,
        "0508_1438": 5.465,
        "0508_1439": 5.464,
        "0508_1440": 5.463,
        "0508_1441": 5.462,
        "0508_1337": 5.461,
        "0508_1341": 5.460,
        "0508_1342": 5.459,
        "0508_1344": 5.458,
        "0508_1345": 5.457,
        "0508_1347": 5.456,
        "0508_1348": 5.455,
        "0508_1349": 5.454,
        "0508_1351": 5.453,
        "0508_1352": 5.452,
        "0508_1354": 5.451,
        "0508_1355": 5.450,
        "0508_1357": 5.449,
        "0508_1358": 5.448,
        "0508_1359": 5.447,
        "0508_1400": 5.446,
        "0508_1401": 5.445,
        "0508_1402": 5.444,
        "0508_1403": 5.443,
        "0508_1404": 5.442,
        "0508_1405": 5.441,
        "0508_1406": 5.436,
        "0508_1407": 5.431,
        "0508_1408": 5.426,
        "0508_1409": 5.421,
        "0508_1411": 5.416,
        "0508_1412": 5.411,
        "0508_1413": 5.406,
        "0508_1415": 5.396,
        "0508_1416": 5.361
    }

    dfreq = dict(sorted(dfreq.items(), key=lambda x: x[1]))
    
    dfile = {}
    for itag,ifreq in dfreq.items():
        ifile = glob.glob(f'May8th/wf_26{itag}*.npz')
        if len(ifile)!=1:
            print('error!!',itag, ifreq)
        dfile[ifreq] = ifile[0]

    pdf0 = PdfPages('pc0.pdf')

    dtbin, dallave_0, dallave_1, dallavecorr_0, dallavecorr_1 = {}, {}, {}, {}, {}
    for ifreq,ifile in dfile.items():
        print(ifile,ifreq)
        res = single_file(ifile,rebin_fac_,pdf0,ifreq)
        dtbin[ifreq] = res['tbin']
        dallave_0[ifreq] = res['all_ave_0']
        dallave_1[ifreq] = res['all_ave_1']
        dallavecorr_0[ifreq] = res['all_ave_corr0']
        dallavecorr_1[ifreq] = res['all_ave_corr1']

    pdf2 = PdfPages('pc2.pdf')

    fig,ax = plt.subplots(nrows=2,sharex=True)
    for idx,ifreq in enumerate(dtbin.keys()):
        ax[0].plot(dtbin[ifreq],dallave_0[ifreq],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
        ax[1].plot(dtbin[ifreq],dallave_1[ifreq],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
    ax[1].set_xlabel('Time [ns]')
    for iax in ax:
        iax.set_ylabel('Voltage [V]')
        iax.set_xlim(-100,1000)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')
    
    fig,ax = plt.subplots(nrows=2,sharex=True)
    for idx,ifreq in enumerate(dtbin.keys()):
        ax[0].plot(dtbin[ifreq],dallavecorr_0[ifreq],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
        ax[1].plot(dtbin[ifreq],dallavecorr_1[ifreq],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
    ax[1].set_xlabel('Time [ns]')
    for iax in ax:
        iax.set_ylabel('Voltage [V]')
        iax.set_xlim(-100,1000)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')

    pdf0.close()
    pdf2.close()
    
    
def sequential_run(rebin_fac_, first_, last_):
    files = glob.glob(f'May8th/*.npz')

    dfile = {}
    for ifile in files:
        ts = re.findall('.*260508_([0-9]*)_.*',ifile)[0]
        ts = int(ts)

        if first_<ts and last_>ts:
            # print(ifile, ts)
            dfile[ts] = ifile

    dfile = dict(sorted(dfile.items(), key=lambda x: x[0]))
    print(dfile)

    pdf0 = PdfPages('pc0.pdf')

    dtbin, dallave_0, dallave_1, dallavecorr_0, dallavecorr_1 = {}, {}, {}, {}, {}
    for its,ifile in dfile.items():
        print(ifile,its)
        res = single_file(ifile,rebin_fac_,pdf0,its)
        dtbin[its] = res['tbin']
        dallave_0[its] = res['all_ave_0']
        dallave_1[its] = res['all_ave_1']
        dallavecorr_0[its] = res['all_ave_corr0']
        dallavecorr_1[its] = res['all_ave_corr1']

    pdf2 = PdfPages('pc2.pdf')

    fig,ax = plt.subplots(nrows=2,sharex=True)
    for idx,its in enumerate(dtbin.keys()):
        ax[0].plot(dtbin[its],dallave_0[its],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
        ax[1].plot(dtbin[its],dallave_1[its],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
    ax[1].set_xlabel('Time [ns]')
    for iax in ax:
        iax.set_ylabel('Voltage [V]')
        iax.set_xlim(-100,1000)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')
    
    fig,ax = plt.subplots(nrows=2,sharex=True)
    for idx,its in enumerate(dtbin.keys()):
        ax[0].plot(dtbin[its],dallavecorr_0[its],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
        ax[1].plot(dtbin[its],dallavecorr_1[its],'-',lw=0.5,c=mpl.cm.jet(idx/len(dtbin)))
    ax[1].set_xlabel('Time [ns]')
    for iax in ax:
        iax.set_ylabel('Voltage [V]')
        iax.set_xlim(-100,1000)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')

    pdf0.close()
    pdf2.close()
    
    
if __name__ == '__main__':
    rebin_fac = int(sys.argv[2])
    if 'npz' in sys.argv[1]:
        single_file(sys.argv[1], rebin_fac)
    elif 'freq_scan'==sys.argv[1]:
        freq_scan_files(rebin_fac)
    elif 'sequential'==sys.argv[1]:
        sequential_run(rebin_fac,int(sys.argv[3]),int(sys.argv[4]))
    

