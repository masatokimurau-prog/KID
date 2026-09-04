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

binped = np.arange(-20,20.001,0.1)
bint0 = np.arange(0,0.2001,0.002)
bink = np.arange(-10,10.001,0.1)
bink2 = np.arange(-10,10.001,0.1)
bintau = np.arange(0,1001e-3,10e-3)
binrise = np.arange(0,161e-3,2e-3)

def main(ddf_, dlivetime_, lblunit_):
    # cls = [mpl.cm.jet(idx/len(ddf_)) for idx in range(len(ddf_))]
    cls = {ifreq:mpl.cm.jet(idx/len(ddf_)) for idx,ifreq in enumerate(ddf_.keys())}
    
    pdf1 = PdfPages('pc1.pdf')
    pdf2 = PdfPages('pc2.pdf')

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
    

def freq_scan(cut_):
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


def single_run(filename_):
    df = pd.read_csv(filename_)
    print(df)

    pdf1 = PdfPages('pc1.pdf')

    lpars = ['ped','t0','k','tau','rise']
    dbins = {'ped':binped, 't0':bint0, 'k':bink, 'tau':bintau, 'rise':binrise}

    llinfit = []
    for ich in range(2):
        linfit = np.polyfit(df[f'ch{ich}_ped'],df[f'ch{ich}_k'],1)
        llinfit.append(np.poly1d(linfit))

    
    ##### 1d hists of each parameter
    fig,ax = plt.subplots(nrows=2,ncols=3)
    ax = ax.flatten()
    for iax,ipar in zip(ax[:len(lpars)],lpars):
        for ich in range(2):
            hy,hx = np.histogram(df[f'ch{ich}_{ipar}'],bins=dbins[ipar])
            iax.hist(hx[:-1],hx,weights=hy,alpha=0.6,label=f'ch{ich}')
        if ipar in ['ped','k']:
            iax.set_xlabel(f'{ipar} [mV]')
        else:
            iax.set_xlabel(f'{ipar} [us]')
        iax.grid()
        iax.legend()
    hy,hx = np.histogram(df['absk'],bins=bink2)
    ax[-1].hist(hx[:-1],hx,weights=hy)
    ax[-1].set_xlabel('|k| [mV]')
    ax[-1].grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    ##### correlation of each parameters btw/ channels
    fig,ax = plt.subplots(nrows=2,ncols=3)
    ax = ax.flatten()
    for iax,ipar in zip(ax[:len(lpars)],lpars):
        h2d,hy,hx = np.histogram2d(df[f'ch0_{ipar}'],df[f'ch1_{ipar}'],bins=(dbins[ipar],dbins[ipar]))
        im = iax.pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=iax)
        if ipar in ['ped','k']:
            iax.set_xlabel(f'ch0 {ipar} [mV]')
            iax.set_ylabel(f'ch1 {ipar} [mV]')
        else:
            iax.set_xlabel(f'ch0 {ipar} [us]')
            iax.set_ylabel(f'ch1 {ipar} [us]')
        iax.grid()
        iax.legend()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    ##### correlations of time constants
    fig,ax = plt.subplots(nrows=2,ncols=2)
    for ich in range(2):
        h2d,hx,hy = np.histogram2d(df[f'ch{ich}_rise'],df[f'ch{ich}_tau'],bins=(binrise,bintau))
        im = ax[ich,0].pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=ax[ich,0])
        ax[ich,0].set_xlabel(f'ch{ich} rise [us]')
        ax[ich,0].set_ylabel(f'ch{ich} tau [us]')

    for iax in ax.flatten():
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    ##### correlations of k and something
    fig,ax = plt.subplots(nrows=2,ncols=2)
    for ich in range(2):
        h2d,hx,hy = np.histogram2d(df[f'ch{ich}_ped'],df[f'ch{ich}_k'],bins=(binped,bink))
        im = ax[ich,0].pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=ax[ich,0])
        ax[ich,0].set_xlabel(f'ch{ich} pedestal [mV]')
        ax[ich,0].set_ylabel(f'ch{ich} k [mV]')
        ax[ich,0].plot(binped,llinfit[ich](binped),'-',c='r',lw=1)

        h2d,hx,hy = np.histogram2d(df[f'ch{ich}_k'],df[f'ch{ich}_rise'],bins=(bink,binrise))
        im = ax[ich,1].pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=ax[ich,1])
        ax[ich,1].set_xlabel(f'ch{ich} k [mV]')
        ax[ich,1].set_ylabel(f'ch{ich} rise [us]')
    for iax in ax.flatten():
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    for ich in range(2):
        ax[ich,0].set_xlim(df[f'ch{ich}_ped'].mean()-2.5,df[f'ch{ich}_ped'].mean()+2.5)
        ax[ich,0].set_ylim(df[f'ch{ich}_k'].mean()-2.5,df[f'ch{ich}_k'].mean()+2.5)
        ax[ich,1].set_xlim(df[f'ch{ich}_k'].mean()-2.5,df[f'ch{ich}_k'].mean()+2.5)
        ax[ich,1].set_ylim(0,0.15)
    fig.savefig(pdf1,format='pdf')
    
    ##### correlations of k and something, after the pedestal correction
    fig,ax = plt.subplots(nrows=2,ncols=2)
    for ich in range(2):
        h2d,hx,hy = np.histogram2d(df[f'ch{ich}_ped'],df[f'ch{ich}_k']*llinfit[ich](df[f'ch{ich}_ped'].mean())/llinfit[ich](df[f'ch{ich}_ped']),bins=(binped,bink))
        im = ax[ich,0].pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=ax[ich,0])
        ax[ich,0].set_xlabel(f'ch{ich} pedestal [mV]')
        ax[ich,0].set_ylabel(f'ch{ich} k corrected [mV]')

        h2d,hx,hy = np.histogram2d(df[f'ch{ich}_k']*llinfit[ich](df[f'ch{ich}_ped'].mean())/llinfit[ich](df[f'ch{ich}_ped']),df[f'ch{ich}_rise'],bins=(bink,binrise))
        im = ax[ich,1].pcolormesh(*np.meshgrid(hx,hy),np.ma.masked_where(h2d.T<=0,h2d.T))
        fig.colorbar(im,ax=ax[ich,1])
        ax[ich,1].set_xlabel(f'ch{ich} k corrected [mV]')
        ax[ich,1].set_ylabel(f'ch{ich} rise [us]')
    for iax in ax.flatten():
        iax.grid()
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')

    for ich in range(2):
        ax[ich,0].set_xlim(df[f'ch{ich}_ped'].mean()-2.5,df[f'ch{ich}_ped'].mean()+2.5)
        ax[ich,0].set_ylim(df[f'ch{ich}_k'].mean()-2.5,df[f'ch{ich}_k'].mean()+2.5)
        ax[ich,1].set_xlim(df[f'ch{ich}_k'].mean()-2.5,df[f'ch{ich}_k'].mean()+2.5)
        ax[ich,1].set_ylim(0,0.15)
    fig.savefig(pdf1,format='pdf')
    

    
    pdf1.close()    
    
    
def sequential_run(first_, last_):
    files = glob.glob(f'May8th/*_fitres.csv')

    df = None
    for ifile in files:
        ts = re.findall('.*260508_([0-9]*)_.*',ifile)[0]
        ts = int(ts)

        if first_<ts and last_>ts:
            print(ifile, ts)

            if df is None: df = pd.read_csv(ifile)
            else: df = pd.concat([df,pd.read_csv(ifile)],axis=0,ignore_index=True)

    print(df)

    df.to_csv('temp.csv',index=False)

    single_run('temp.csv')

    os.remove('temp.csv')
    

    
if __name__ == '__main__':
    sw = int(sys.argv[1])
    common_cut = '0<ch0_tau<0.99 and 0<ch1_tau<0.99 and -50<ch0_k<50'
    match sw:
        case 1:
            single_run(sys.argv[2])
        case 2:
            freq_scan(common_cut)
        case 3:
            sequential_run(int(sys.argv[2]),int(sys.argv[3]))
        case _:
            print('invalid switch of',sw)


