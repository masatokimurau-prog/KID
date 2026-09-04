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

import datetime as dt
from scipy.optimize import least_squares

import ds_style

# param_keys_free = ['a','Ql','Qc','fr']
# param_keys_fix = {'alpha':0,'tau':0,'phi':0}
param_keys_free = ['a','Ql','Qc','fr','phi','alpha','tau']
param_keys_fix = {}

def dict_to_array(d):
    return np.array([d[k] for k in param_keys_free])

def array_to_dict(arr):
    d = {k: v for k, v in zip(param_keys_free, arr)}
    d.update(param_keys_fix)
    return d

def S21notch(par_, f_):
    env = par_['a']*np.exp(1j*par_['alpha'])*np.exp(-2j*np.pi*f_*par_['tau'])
    res = (par_['Ql']/np.abs(par_['Qc'])*np.exp(1j*par_['phi'])) / (1+2j*par_['Ql']*(f_/par_['fr']-1))
    return env*(1-res)

def residual_S21notch(par_, x_, y_):
    par = array_to_dict(par_)
    pred = S21notch(par, x_)
    return np.concatenate([np.real(pred-y_),np.imag(pred-y_)])

def main():
    files = glob.glob('../daqpc_copy/s2p_20251113/*.s2p')

    ##### read data
    ddf = {}
    dfiles = {}
    for ifile in files:
        dfi = pd.read_csv(ifile, skiprows=1, comment='!', delim_whitespace=True, names=['F','S11','P11','S12','P12','S21','P21','S22','P22'])
        # print(dfi)
        temp = re.findall('[0-9]*_([0-9]*\.[0-9]*)K',ifile)
        print(temp[0])

        dfi['F'] = dfi['F']/1e9

        ddf[float(temp[0])] = dfi
        dfiles[float(temp[0])] = ifile

    ddf = dict(sorted(ddf.items()))
    print(ddf.keys())
    df_bg = list(ddf.values())[-1]
    t_bg = list(ddf.keys())[-1]

    ##### fit to data
    dfit = {}
    dfitres = {}
    # fit_rng_f = [5.42,5.52]
    # fit_cut = f'{fit_rng_f[0]}<=F<={fit_rng_f[1]}'
    dfit_rng_f = {}
    for iT, idf in ddf.items():
        for icol in idf.columns:            
            idf[f'{icol}_subt'] = idf[icol] - df_bg[icol]
            # idf[f'{icol}_subt'] = idf[icol]
            if icol[0]=='P':
                idf[f'{icol}_subt'] = (idf[f'{icol}_subt'] + 180) % (360) - 180

        for icol in ['11','21','12','22']:
            idf[f'A{icol}'] = 10**(idf[f'S{icol}_subt']/20)
            idf[f'Cx{icol}'] = idf[f'A{icol}'] * np.exp(1j*idf[f'P{icol}_subt']*np.pi/180)
            idf[f'Re{icol}'] = idf[f'A{icol}'] * np.cos(idf[f'P{icol}_subt']*np.pi/180)
            idf[f'Im{icol}'] = idf[f'A{icol}'] * np.sin(idf[f'P{icol}_subt']*np.pi/180)

        f_min = idf.loc[np.abs(idf['Cx21']).idxmin(),'F']
        init = {'a':1.0,'Qc':300,'Ql':300,'fr':f_min,'phi':0,'alpha':0,'tau':0}
        # init = {'a':1.0e4,'Qc':300,'Ql':300,'fr':f_min,'phi':0,'alpha':200,'tau':2.0}
        fit_rng_f = [f_min-0.025,f_min+0.025]
        fit_cut = f'{fit_rng_f[0]}<=F<={fit_rng_f[1]}'
        idf_cut = idf.query(fit_cut)
        res = least_squares(residual_S21notch, dict_to_array(init), args=(idf_cut.F,idf_cut.Cx21))

        dfit_rng_f[iT] = fit_rng_f
        dfit[iT] = res
        dfitres[iT] = S21notch(array_to_dict(res.x),idf_cut.F)
        

    ##### draw canvas
    pdf1 = PdfPages('pc1.pdf')
    pdf2 = PdfPages('pc2.pdf')
    pdf3 = PdfPages('pc3.pdf')

    leg_opt = {'columnspacing':1,'fontsize':'small','handlelength':1}
    
    for iT, idf in ddf.items():
        fig,ax = plt.subplots(figsize=(16,9),nrows=2,ncols=4,sharex=True,sharey='row')
        ax[0,0].plot(idf.F,idf.S11,'.',label=f'S11, T={iT:.1f} K')
        ax[0,0].plot(df_bg.F,df_bg.S11,'.',label=f'S11, T={t_bg:.1f} K')
        ax[0,1].plot(idf.F,idf.S21,'.',label=f'S21, T={iT:.1f} K')
        ax[0,1].plot(df_bg.F,df_bg.S21,'.',label=f'S21, T={t_bg:.1f} K')
        ax[0,2].plot(idf.F,idf.S12,'.',label=f'S12, T={iT:.1f} K')
        ax[0,2].plot(df_bg.F,df_bg.S12,'.',label=f'S12, T={t_bg:.1f} K')
        ax[0,3].plot(idf.F,idf.S22,'.',label=f'S22, T={iT:.1f} K')
        ax[0,3].plot(df_bg.F,df_bg.S22,'.',label=f'S22, T={t_bg:.1f} K')
        
        ax[1,0].plot(idf.F,idf.P11,'.',label=f'P11, T={iT:.1f} K')
        ax[1,0].plot(df_bg.F,df_bg.P11,'.',label=f'P11, T={t_bg:.1f} K')
        ax[1,1].plot(idf.F,idf.P21,'.',label=f'P21, T={iT:.1f} K')
        ax[1,1].plot(df_bg.F,df_bg.P21,'.',label=f'P21, T={t_bg:.1f} K')
        ax[1,2].plot(idf.F,idf.P12,'.',label=f'P12, T={iT:.1f} K')
        ax[1,2].plot(df_bg.F,df_bg.P12,'.',label=f'P12, T={t_bg:.1f} K')
        ax[1,3].plot(idf.F,idf.P22,'.',label=f'P22, T={iT:.1f} K')
        ax[1,3].plot(df_bg.F,df_bg.P22,'.',label=f'P22, T={t_bg:.1f} K')
        
        for iax in ax.flatten():
            iax.grid()
            iax.legend()
        for idx in range(4):
            ax[1,idx].set_xlabel('Frequency [GHz]')
        ax[0,0].set_ylabel('Amplitude [dB]')
        ax[1,0].set_ylabel('Phase [deg]')
        
        fig.tight_layout()
        fig.savefig(pdf1,format='pdf')
    
    fig,ax = plt.subplots(figsize=(16,9),nrows=2,ncols=4,sharex=True,sharey='row')
    for iT, idf in ddf.items():
        if iT==t_bg: continue
        ax[0,0].plot(idf.F,idf.S11_subt,'.',label=f'{iT:.1f} K')
        ax[0,1].plot(idf.F,idf.S21_subt,'.',label=f'{iT:.1f} K')
        ax[0,2].plot(idf.F,idf.S12_subt,'.',label=f'{iT:.1f} K')
        ax[0,3].plot(idf.F,idf.S22_subt,'.',label=f'{iT:.1f} K')
        
        ax[1,0].plot(idf.F,idf.P11_subt,'.',label=f'{iT:.1f} K')
        ax[1,1].plot(idf.F,idf.P21_subt,'.',label=f'{iT:.1f} K')
        ax[1,2].plot(idf.F,idf.P12_subt,'.',label=f'{iT:.1f} K')
        ax[1,3].plot(idf.F,idf.P22_subt,'.',label=f'{iT:.1f} K')
        
    ax[0,0].legend(title='S11',ncols=2,**leg_opt)
    ax[0,1].legend(title='S21',ncols=2,columnspacing=1,fontsize='small')
    ax[0,2].legend(title='S12',ncols=2,columnspacing=1,fontsize='small')
    ax[0,3].legend(title='S22',ncols=2,columnspacing=1,fontsize='small')
        
    ax[1,0].legend(title='P11',ncols=2,columnspacing=1,fontsize='small')
    ax[1,1].legend(title='P21',ncols=2,columnspacing=1,fontsize='small')
    ax[1,2].legend(title='P12',ncols=2,columnspacing=1,fontsize='small')
    ax[1,3].legend(title='P22',ncols=2,columnspacing=1,fontsize='small')

    for iax in ax.flatten():
        iax.grid()

    for idx in range(4):
        ax[1,idx].set_xlabel('Frequency [GHz]')
    ax[0,0].set_ylabel('Amplitude [dB]')
    ax[1,0].set_ylabel('Phase [deg]')
    fig.tight_layout()
    fig.savefig(pdf1,format='pdf')


    for iT, idf in ddf.items():
        if iT==t_bg: continue
        fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=2,sharex=True,sharey=True)
        ax[0,0].plot(idf.Re11,idf.Im11,'.',label=f'S11, T={iT:.1f} K')
        ax[0,1].plot(idf.Re21,idf.Im21,'.',label=f'S21, T={iT:.1f} K')
        ax[1,0].plot(idf.Re12,idf.Im12,'.',label=f'S12, T={iT:.1f} K')
        ax[1,1].plot(idf.Re22,idf.Im22,'.',label=f'S22, T={iT:.1f} K')

        for iax in ax.flatten():
            iax.grid()
            iax.legend()

        for idx in range(2):
            ax[idx,0].set_ylabel('Im')
            ax[1,idx].set_xlabel('Re')

        fig.tight_layout()
        fig.savefig(pdf2,format='pdf')


    fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=2,sharex=True,sharey=True)        
    for iT, idf in ddf.items():
        if iT==t_bg: continue
        ax[0,0].plot(idf.Re11,idf.Im11,'.',label=f'{iT:.1f} K',alpha=0.5)
        ax[0,1].plot(idf.Re21,idf.Im21,'.',label=f'{iT:.1f} K',alpha=0.5)
        ax[1,0].plot(idf.Re12,idf.Im12,'.',label=f'{iT:.1f} K',alpha=0.5)
        ax[1,1].plot(idf.Re22,idf.Im22,'.',label=f'{iT:.1f} K',alpha=0.5)

    ax[0,0].legend(title='S11',ncols=2,columnspacing=1,fontsize='small')
    ax[0,1].legend(title='S21',ncols=2,columnspacing=1,fontsize='small')
    ax[1,0].legend(title='S12',ncols=2,columnspacing=1,fontsize='small')
    ax[1,1].legend(title='S22',ncols=2,columnspacing=1,fontsize='small')

    for iax in ax.flatten():
        iax.grid()

    for idx in range(2):
        ax[idx,0].set_ylabel('Im')
        ax[1,idx].set_xlabel('Re')
    fig.tight_layout()
    fig.savefig(pdf2,format='pdf')

    nr, nc = 2, 3

    fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
    ax = ax.flatten()
    for idx,iT in enumerate([x for x in ddf.keys() if x!=t_bg]):
        fitres = array_to_dict(dfit[iT].x)
        ax[idx].plot(ddf[iT].Re21,ddf[iT].Im21,'.',label=f'Data ({iT:.1f} K)')
        ax[idx].plot(np.real(dfitres[iT]),np.imag(dfitres[iT]),'-',label=f'Fit\na={fitres["a"]:.2f}, Ql={fitres["Ql"]:.0f}, Qc={fitres["Qc"]:.0f}, \nfr={fitres["fr"]:.4f}, phi={fitres["phi"]:.2f}\nalpha={fitres["alpha"]:.2f}, tau={fitres["tau"]:.2f}')
        ax[idx].grid()
        ax[idx].legend(**leg_opt,loc='upper left')
    for idx in range(nc):
        ax[nc*(nr-1)+idx].set_xlabel('Re')
    for idx in range(nr):
        ax[idx*nc].set_ylabel('Im')        
    fig.tight_layout()
    fig.savefig(pdf3,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
    ax = ax.flatten()
    for idx,iT in enumerate([x for x in ddf.keys() if x!=t_bg]):
        fitres = array_to_dict(dfit[iT].x)
        ax[idx].plot(ddf[iT].F,np.abs(ddf[iT].Cx21),'.',label=f'Data ({iT:.1f} K)')
        fit_cut = f'{dfit_rng_f[iT][0]}<=F<={dfit_rng_f[iT][1]}'
        ax[idx].plot(ddf[iT].query(fit_cut).F,np.abs(dfitres[iT]),'-',label=f'Fit\na={fitres["a"]:.2f}, Ql={fitres["Ql"]:.0f}, Qc={fitres["Qc"]:.0f}, \nfr={fitres["fr"]:.4f}, phi={fitres["phi"]:.2f}\nalpha={fitres["alpha"]:.2f}, tau={fitres["tau"]:.2f}')
        ax[idx].grid()
        ax[idx].legend(**leg_opt,loc='lower left')
    for idx in range(nc):
        ax[nc*(nr-1)+idx].set_xlabel('Frequency [GHz]')
    for idx in range(nr):
        ax[idx*nc].set_ylabel('|S21|')        
    fig.tight_layout()
    fig.savefig(pdf3,format='pdf')

    fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
    ax = ax.flatten()
    for idx,iT in enumerate([x for x in ddf.keys() if x!=t_bg]):
        fitres = array_to_dict(dfit[iT].x)
        ax[idx].plot(ddf[iT].F,np.angle(ddf[iT].Cx21),'.',label=f'Data ({iT:.1f} K)')
        fit_cut = f'{dfit_rng_f[iT][0]}<=F<={dfit_rng_f[iT][1]}'
        ax[idx].plot(ddf[iT].query(fit_cut).F,np.angle(dfitres[iT]),'-',label=f'Fit\na={fitres["a"]:.2f}, Ql={fitres["Ql"]:.0f}, Qc={fitres["Qc"]:.0f}, \nfr={fitres["fr"]:.4f}, phi={fitres["phi"]:.2f}\nalpha={fitres["alpha"]:.2f}, tau={fitres["tau"]:.2f}')
        ax[idx].grid()
        ax[idx].legend(**leg_opt,loc='lower left')
    for idx in range(nc):
        ax[nc*(nr-1)+idx].set_xlabel('Frequency [GHz]')
    for idx in range(nr):
        ax[idx*nc].set_ylabel('Phase [rad]')
    fig.tight_layout()
    fig.savefig(pdf3,format='pdf')

    fig,ax = plt.subplots()
    lT, lQl, lQc, lQi = [], [], [], []
    for iT,ifit in dfit.items():
        if iT==t_bg: continue
        lT.append(iT)
        fitres = array_to_dict(ifit.x)
        lQl.append(fitres['Ql'])
        lQc.append(fitres['Qc'])
        lQi.append(1/(1/fitres['Ql']-1/fitres['Qc']))
   
    ax.errorbar(x=lT, y=lQl, fmt='o', label='Ql')
    ax.errorbar(x=lT, y=lQc, fmt='o', label='Qc')
    ax.errorbar(x=lT, y=lQi, fmt='o', label='Qi')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Q')
    ax.set_yscale('log')
    fig.savefig('pc4.pdf')
    
    pdf1.close()
    pdf2.close()
    pdf3.close()


    ##### resonator tool
    from resonator_tools import circuit
    def main_restool():
        dport = {}
        for iT,ifile in dfiles.items():
            port1 = circuit.notch_port()
            port1.add_froms2p(ifile,3,4,'dBmagphasedeg',fdata_unit=1e0,delimiter=None)
            print(dfit_rng_f[iT])
            # print(port1.f_data)
            # print(np.logical_and(port1.f_data>dfit_rng_f[iT][0],port1.f_data<dfit_rng_f[iT][1]))
            port1.autofit(fcrop=(dfit_rng_f[iT][0]*1e9,dfit_rng_f[iT][1]*1e9))#, electric_delay=2.0e-9)
            print(iT)
            print(port1._delay,port1.fitresults)

            dport[iT] = port1


        dport = dict(sorted(dport.items()))
        pdf11 = PdfPages('pc11.pdf')

        fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
        ax = ax.flatten()
        for idx,(iT,iport) in enumerate(dport.items()):
            if idx>=nr*nc: break
            fid = np.logical_and(port1.f_data>=dfit_rng_f[iT][0]*1e9,port1.f_data<=dfit_rng_f[iT][1]*1e9)
            re = iport.z_data_raw.real
            im = iport.z_data_raw.imag
            re2 = iport.z_data_sim.real
            im2 = iport.z_data_sim.imag

            ax[idx].plot(re,im,'.',label=f'Data ({iT:.1f} K)')
            ax[idx].plot(re2[fid],im2[fid],'-',label='Fit')
            ax[idx].grid()
            ax[idx].legend(**leg_opt,loc='upper left')
        for idx in range(nc):
            ax[nc*(nr-1)+idx].set_xlabel('Re')
        for idx in range(nr):
            ax[idx*nc].set_ylabel('Im')        
        fig.tight_layout()
        fig.savefig(pdf11,format='pdf')

        fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
        ax = ax.flatten()
        for idx,(iT,iport) in enumerate(dport.items()):
            if idx>=nr*nc: break
            fid = np.logical_and(port1.f_data>=dfit_rng_f[iT][0]*1e9,port1.f_data<=dfit_rng_f[iT][1]*1e9)
            fq = iport.f_data*1e-9
            ab = np.absolute(iport.z_data_raw)
            ab2 = np.absolute(iport.z_data_sim)

            ax[idx].plot(fq,ab,'.',label=f'Data ({iT:.1f} K)')
            ax[idx].plot(fq[fid],ab2[fid],'-',label='Fit')
            ax[idx].grid()
            ax[idx].legend(**leg_opt,loc='lower left')
        for idx in range(nc):
            ax[nc*(nr-1)+idx].set_xlabel('Frequency [GHz]')
        for idx in range(nr):
            ax[idx*nc].set_ylabel('|S21|')
        fig.tight_layout()
        fig.savefig(pdf11,format='pdf')
        
        fig,ax = plt.subplots(figsize=(16,9),nrows=nr,ncols=nc,sharex=True,sharey=True)
        ax = ax.flatten()
        for idx,(iT,iport) in enumerate(dport.items()):
            if idx>=nr*nc: break
            fid = np.logical_and(port1.f_data>=dfit_rng_f[iT][0]*1e9,port1.f_data<=dfit_rng_f[iT][1]*1e9)
            fq = iport.f_data*1e-9
            ph = np.angle(iport.z_data_raw)
            ph2 = np.angle(iport.z_data_sim)

            ax[idx].plot(fq,ph,'.',label=f'Data ({iT:.1f} K)')
            ax[idx].plot(fq[fid],ph2[fid],'-',label='Fit')
            ax[idx].grid()
            ax[idx].legend(**leg_opt,loc='lower left')
        for idx in range(nc):
            ax[nc*(nr-1)+idx].set_xlabel('Frequency [GHz]')
        for idx in range(nr):
            ax[idx*nc].set_ylabel('Phase [rad]')
        fig.tight_layout()
        fig.savefig(pdf11,format='pdf')
        
        pdf11.close()

    main_restool()
       

    
if __name__ == '__main__':
    main()

