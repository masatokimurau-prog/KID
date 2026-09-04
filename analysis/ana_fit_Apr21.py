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

def main(ddf_, dlivetime_, lblunit_):
    # cls = [mpl.cm.jet(idx/len(ddf_)) for idx in range(len(ddf_))]
    cls = {ifreq:mpl.cm.jet(idx/len(ddf_)) for idx,ifreq in enumerate(ddf_.keys())}
    
    pdf1 = PdfPages('pc1.pdf')
    pdf2 = PdfPages('pc2.pdf')

    binped = np.arange(-20,20.001,0.5)
    bink = np.arange(0,20.001,0.2)
    bink2 = np.arange(-15,10.001,0.4)
    bintau = np.arange(0,1001e-3,10e-3)
    binrise = np.arange(0,161e-3,2e-3)

    leg_opt = {'columnspacing':1,'fontsize':'x-small','handlelength':1,'ncols':2}

    fig,ax = plt.subplots(ncols=2,nrows=2)
    ax = ax.flatten()
    err_opt = {'capsize':2, 'elinewidth':1, 'markersize':0.5,'fmt':'.'}
    for ifreq,idf in ddf_.items():
        h,_ = np.histogram(idf.ch1_ped,bins=binped)
        ax[0].hist(binped[:-1],binped,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
        ax[0].errorbar(x=0.5*(binped[1:]+binped[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
        h,_ = np.histogram(idf.absk,bins=bink)
        ax[1].hist(bink[:-1],bink,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
        ax[1].errorbar(x=0.5*(bink[1:]+bink[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
        h,_ = np.histogram(idf.ch0_k,bins=bink2)
        ax[2].hist(bink2[:-1],bink2,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
        ax[2].errorbar(x=0.5*(bink2[1:]+bink2[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
        h,_ = np.histogram(idf.ch1_k,bins=bink2)
        ax[3].hist(bink2[:-1],bink2,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
        ax[3].errorbar(x=0.5*(bink2[1:]+bink2[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
    ax[0].set_xlabel('Ch1 pedestal [mV]')
    ax[0].set_ylim(0,0.3)
    ax[1].set_xlabel('|K| [mV]')
    ax[1].set_ylim(0,0.13)
    ax[2].set_xlabel('Ch0 k [mV]')
    ax[2].set_ylim(0,0.25)
    ax[3].set_xlabel('Ch1 k [mV]')
    ax[3].set_ylim(0,0.4)
    for iax in ax:
        iax.set_ylabel('Ev/bin/sec')
        iax.legend(**leg_opt)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots(ncols=2,nrows=2)
    ax = ax.flatten()
    for ich in range(2):
        for ifreq,idf in ddf_.items():
            h,_ = np.histogram(idf[f'ch{ich}_tau'],bins=bintau)
            ax[ich].hist(bintau[:-1],bintau,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
            ax[ich].errorbar(x=0.5*(bintau[1:]+bintau[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
            h,_ = np.histogram(idf[f'ch{ich}_rise'],bins=binrise)
            ax[ich+2].hist(binrise[:-1],binrise,weights=h/dlivetime_[ifreq],histtype='step',color=cls[ifreq])
            ax[ich+2].errorbar(x=0.5*(binrise[1:]+binrise[:-1]),y=h/dlivetime_[ifreq],yerr=np.sqrt(h)/dlivetime_[ifreq],color=cls[ifreq],label=f'{ifreq} {lblunit_}',**err_opt)
        ax[ich].set_xlabel(f'Ch{ich} tau [us]')
        ax[ich+2].set_xlabel(f'Ch{ich} rise [us]')
    ax[2].set_ylim(0,0.3)
    ax[3].set_ylim(0,0.15)
    for iax in ax:
        iax.set_ylabel('Ev/bin/sec')
        iax.legend(**leg_opt)
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    fig,ax = plt.subplots()
    for ifreq,idf in ddf_.items():
        ax.plot(idf.ch0_k,idf.ch0_rise,'.',c=cls[ifreq],alpha=0.2)
    ax.set_xlabel('Ch0 k [mV]')
    ax.set_ylabel('Ch0 rise [us]')
    ax.set_ylim(0,0.2)
    ax.grid()
    fig.savefig(pdf2,format='pdf')

    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_k,idf.ch0_rise,bins=(bink2,binrise))
        im = ax[idx].pcolormesh(*np.meshgrid(bink2,binrise),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(bink2[1],binrise[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 rise [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 k [mV]')
        ax[4+idx].set_xlabel('ch0 k [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_k,idf.ch0_tau,bins=(bink2,bintau))
        im = ax[idx].pcolormesh(*np.meshgrid(bink2,bintau),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(bink2[1],bintau[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 tau [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 k [mV]')
        ax[4+idx].set_xlabel('ch0 k [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_tau,idf.ch0_rise,bins=(bintau,binrise))
        im = ax[idx].pcolormesh(*np.meshgrid(bintau,binrise),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(bintau[1],binrise[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 rise [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 tau [us]')
        ax[4+idx].set_xlabel('ch0 tau [us]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch1_ped,idf.ch0_k,bins=(binped,bink2))
        im = ax[idx].pcolormesh(*np.meshgrid(binped,bink2),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(binped[1],bink2[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 k [mV]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch1 pedestal [mV]')
        ax[4+idx].set_xlabel('ch1 pedestal [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch1_ped,idf.ch0_tau,bins=(binped,bintau))
        im = ax[idx].pcolormesh(*np.meshgrid(binped,bintau),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(binped[1],bintau[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 tau [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch1 pedestal [mV]')
        ax[4+idx].set_xlabel('ch1 pedestal [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch1_ped,idf.ch0_rise,bins=(binped,binrise))
        im = ax[idx].pcolormesh(*np.meshgrid(binped,binrise),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(binped[1],binrise[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch0 rise [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch1 pedestal [mV]')
        ax[4+idx].set_xlabel('ch1 pedestal [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_k,idf.ch1_k,bins=(bink2,bink2))
        im = ax[idx].pcolormesh(*np.meshgrid(bink2,bink2),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(bink2[1],bink2[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch1 k [mV]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 k [mV]')
        ax[4+idx].set_xlabel('ch0 k [mV]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_rise,idf.ch1_rise,bins=(binrise,binrise))
        im = ax[idx].pcolormesh(*np.meshgrid(binrise,binrise),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(binrise[1],binrise[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch1 r [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 r [us]')
        ax[4+idx].set_xlabel('ch0 r [us]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=False,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        h,_,_ = np.histogram2d(idf.ch0_tau,idf.ch1_tau,bins=(bintau,bintau))
        im = ax[idx].pcolormesh(*np.meshgrid(bintau,bintau),np.ma.masked_where(h.T<=0,h.T))
        fig.colorbar(im,ax=ax[idx])
        ax[idx].text(bintau[1],bintau[-4],f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch1 tau [us]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('ch0 tau [us]')
        ax[4+idx].set_xlabel('ch0 tau [us]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    fig,ax = plt.subplots(figsize=(16,8),ncols=4,nrows=2,sharex=True,sharey=True)
    ax = ax.flatten()
    for idx,(ifreq,idf) in enumerate(ddf_.items()):
        ax[idx].plot(idf.dt_ms%1000,idf.ch1_ped,'.')
        ax[idx].text(0,10,f'{ifreq} {lblunit_}',ha='left',va='top')
        ax[idx].grid()
    for idx in range(2):
        ax[4*idx].set_ylabel('ch1 pedestal [mV]')
    for idx in range(4):
        if len(ddf_)<8: ax[idx].set_xlabel('Time Modulo [ms]')
        ax[4+idx].set_xlabel('Time Modulo [ms]')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')    
    
    
    pdf1.close()
    pdf2.close()
    

def freq_scan_38mm(cut_):
    dfreq = {1309:5.495, 1338:5.494, 1350:5.493, 1357:5.492, 1408:5.491, 1416:5.490, 1430:5.489}#, 1449:5.488}, 1508:5.489}
    ddf = {}
    dlivetime = {}

    for itag,ifreq in dfreq.items():
        ifile = glob.glob(f'./April21st/wf_260422_{itag}*_fitres.csv')
        if len(ifile)==0: continue
        idf = pd.read_csv(ifile[0])
        idf = idf.query(cut_)
        ddf[ifreq] = idf

        nev = 500
        if len(idf)<nev*0.8 or len(idf)>nev*1.1:
            print('nev may be wrong?',nev,len(idf))
            sys.exit()
        
        rate = float(re.match(r'.*_([0-9]+\.[0-9]+)Hz.*',ifile[0]).groups()[0])
        dlivetime[ifreq] = nev/rate
        
    main(ddf,dlivetime,'GHz')

    
def length_scan_5492MHz(cut_):
    dlength = {1357:38, 1547:33, 1719:28}
    ddf = {}
    dlivetime = {}

    for itag,ilg in dlength.items():
        ifile = glob.glob(f'./April21st/wf_260422_{itag}*_fitres.csv')
        if len(ifile)==0: continue
        idf = pd.read_csv(ifile[0])
        idf = idf.query(cut_)
        ddf[ilg] = idf

        nev = 500
        if len(idf)<nev*0.9 or len(idf)>nev*1.1:
            print('nev may be wrong?',nev,len(idf))
            sys.exit()
            
        rate = float(re.match(r'.*_([0-9]+\.[0-9]+)Hz.*',ifile[0]).groups()[0])
        dlivetime[ilg] = nev/rate
        
    main(ddf,dlivetime,'mm')


def length_scan_5491MHz(cut_):
    dlength = {1408:38, 1527:33, 1727:28, 1849:23}
    ddf = {}
    dlivetime = {}

    for itag,ilg in dlength.items():
        ifile = glob.glob(f'./April21st/wf_260422_{itag}*_fitres.csv')
        if len(ifile)==0: continue
        print(ifile[0])
        idf = pd.read_csv(ifile[0])
        idf = idf.query(cut_)
        ddf[ilg] = idf

        nev = 500
        if len(idf)<nev*0.9 or len(idf)>nev*1.1:
            print('nev may be wrong?',nev,len(idf))
            sys.exit()
            
        rate = float(re.match(r'.*_([0-9]+\.[0-9]+)Hz.*',ifile[0]).groups()[0])
        dlivetime[ilg] = nev/rate

    main(ddf,dlivetime,'mm')


def long_run(cut_):
    lfile = ['April21st/wf_260422_113811_0.32Hz.npz', 'April21st/wf_260422_104339_0.31Hz.npz', 'April21st/wf_260422_074954_0.42Hz.npz', 'April21st/wf_260422_071127_0.43Hz.npz', 'April21st/wf_260422_033222_0.46Hz.npz', 'April21st/wf_260422_091324_0.38Hz.npz', 'April21st/wf_260422_055913_0.46Hz.npz', 'April21st/wf_260422_082923_0.38Hz.npz', 'April21st/wf_260422_052213_0.45Hz.npz', 'April21st/wf_260422_012037_0.52Hz.npz', 'April21st/wf_260422_040816_0.46Hz.npz', 'April21st/wf_260422_025510_0.45Hz.npz', 'April21st/wf_260422_015238_0.51Hz.npz', 'April21st/wf_260422_095654_0.36Hz.npz', 'April21st/wf_260422_022511_0.56Hz.npz', 'April21st/wf_260422_063505_0.46Hz.npz', 'April21st/wf_260422_001206_0.46Hz.npz', 'April21st/wf_260422_044424_0.44Hz.npz', 'April21st/wf_260422_004801_0.51Hz.npz', 'April21st/wf_260421_175023_0.42Hz.npz', 'April21st/wf_260421_183008_0.47Hz.npz', 'April21st/wf_260421_190533_0.48Hz.npz', 'April21st/wf_260421_194030_0.50Hz.npz', 'April21st/wf_260421_230339_0.48Hz.npz', 'April21st/wf_260421_212355_0.49Hz.npz', 'April21st/wf_260421_204918_0.48Hz.npz', 'April21st/wf_260421_201411_0.48Hz.npz', 'April21st/wf_260421_233829_0.50Hz.npz', 'April21st/wf_260421_215743_0.52Hz.npz', 'April21st/wf_260421_223004_0.50Hz.npz']

    ddf = {}
    dtime = {}
    dlivetime = {}
    for ifile in lfile:
        itime = re.match(r'.*_(26042[1-2]_[0-9]+)_.*',ifile).groups()[0]

        ifile = ifile.replace('.npz','_fitres.csv')

        idf = pd.read_csv(ifile)
        idf = idf.query(cut_)
        ddf[itime] = idf

        nev = 1000
        if len(idf)<nev*0.9 or len(idf)>nev*1.1:
            print('nev may be wrong?',nev,len(idf))
            sys.exit()
        
        rate = float(re.match(r'.*_([0-9]+\.[0-9]+)Hz.*',ifile).groups()[0])
        dlivetime[itime] = nev/rate

        print(itime,rate)

    main(ddf,dlivetime,'')
    
        
if __name__ == '__main__':
    sw = int(sys.argv[1])
    common_cut = '0<ch0_tau<0.99 and 0<ch1_tau<0.99 and -50<ch0_k<50'
    match sw:
        case 1:
            freq_scan_38mm(common_cut)
        case 2:
            length_scan_5492MHz(common_cut)
        case 3:
            length_scan_5491MHz(common_cut)
        case 4:
            long_run(common_cut)
        case _:
            print('invalid switch of',sw)


