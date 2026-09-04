import os
import re
import sys
import time
import glob
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  
import quadpy

import ds_style

MYVERBOSE_ = False

def S21notch(par_, f_):
    env = par_['a']*np.exp(1j*par_['alpha'])*np.exp(-1j*2*np.pi*f_*par_['tau'])
    res = (par_['Ql']/np.abs(par_['Qc'])*np.exp(1j*par_['phi'])) / (1+2*1j*par_['Ql']*(f_/par_['fr']-1))
    return env*(1-res)

def fr_T(T_):
    fr_0 = 5.4e9
    Tc = 9.2
    alpha = 0.05
    return fr_0 / np.sqrt(1-alpha+alpha/(1-(T_/Tc)**4))

def Ql_T(T_):
    val = 2.06e3 -1.28e2*T_ -1.47e1*T_**2
    return np.where(val>1.0, val, 1.0)

def Qc_T(T_):
    return 1.6e3 + 3.5e2*np.log(1+np.exp((T_-6.2)/0.35))

def funcfit(X_, a_, tau_, c_):
    return a_*np.exp(-X_/tau_) + c_

def main(Kinit_, Klast_):
    Kinit = Kinit_
    Klast = Klast_
    # Nominal_par = {'a':1.0, 'alpha':0, 'tau':0e-9, 'Ql':300, 'Qc':299, 'phi':0, 'fr':5.4e9}
    nominal_par = {'a':1.0, 'alpha':0, 'tau':0e-9, 'Ql':Ql_T(Klast), 'Qc':Qc_T(Klast), 'phi':0, 'fr':5.4e9}

    tau_r,tau_d = 50, 200
    t0 = 0
    tmax = np.log(tau_r/tau_d)/(1/tau_d-1/tau_r)
    lT = np.arange(1000) ## time
    # lK = (Klast-Kinit)/(lT[-1]-lT[0])*lT + 6.0 ## temp
    lK = Klast + (Kinit-Klast)*(-np.exp(-(lT-t0)/tau_r) + np.exp(-(lT-t0)/tau_d))
    # lK =  (Kinit-Klast)*np.exp(-lT/100)+Klast ## temp
                  
    lfr = fr_T(lK)
    # print(lT,lfr)

    if False:
        lQl = Ql_T(lK)
        lQc = Qc_T(lK)
    else:
        lQl = np.full(len(lK),nominal_par['Ql'])
        lQc = np.full(len(lK),nominal_par['Qc'])

    lo_freq = 5.3865e9-0.0065e9
    lS21 = []
    for ifr,iQl,iQc in zip(lfr,lQl,lQc):
        pars = copy.deepcopy(nominal_par)
        pars['fr'] = ifr
        pars['Ql'] = iQl
        pars['Qc'] = iQc
        lS21.append(S21notch(pars, lo_freq))

    lS21 = np.array(lS21)
    # lS21 = lS21 * np.exp(1j*np.pi/3)

    fig,ax = plt.subplots(figsize=(9,9),ncols=2,nrows=4,sharex=True)
    ax[0,0].plot(lT,lK,'-')
    # ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Temperature')
    
    # ax[0,1].plot(lS21.real,lS21.imag,'-')
    # ax[0,1].set_xlabel('S21 Real')
    # ax[0,1].set_ylabel('S21 Imag')
    ax[0,1].plot(lT,lfr,'-')
    # ax[0,1].set_xlabel('Time')
    ax[0,1].set_ylabel('Resonant Freq')

    ax[1,0].plot(lT,lQl,'-',label='Ql')
    ax[1,0].plot(lT,lQc,'-',label='Qc')
    # ax[1,0].plot(lT,1/(1/lQl-1/lQc),'-',label='Qi')
    ax[1,0].set_ylabel('Q')
    
    # ax[1,1].plot(lT,lQl,'-',label='Ql')
    # ax[1,1].plot(lT,lQc,'-',label='Qc')
    ax[1,1].plot(lT,1/(1/lQl-1/lQc),'-',label='Qi')
    ax[1,1].set_ylabel('Q')
    # ax[1,1].set_yscale('log')
    
    ax[2,0].plot(lT,lS21.real,'-',label='Real')
    fitr = np.where(lT>tmax*2)
    popt,pconv = curve_fit(funcfit, lT[fitr], lS21[fitr].real, sigma=0.01, p0=[(lS21.real[-1]-lS21[fitr].real[0]),tau_d,lS21.real[-1]])
    ax[2,0].plot(lT[fitr],funcfit(lT[fitr],*popt),'-',label=f"tau = {popt[1]:.1f}",lw=1)
    # ax[1,0].set_xlabel('Time')
    ax[2,0].set_ylabel('S21 Real')
    
    ax[2,1].plot(lT,lS21.imag,'-',label='Imag')
    popt,pconv = curve_fit(funcfit, lT[fitr], lS21[fitr].imag, sigma=0.01, p0=[(lS21.imag[-1]-lS21[fitr].imag[0]),tau_d,lS21.imag[-1]])
    ax[2,1].plot(lT[fitr],funcfit(lT[fitr],*popt),'-',label=f"tau = {popt[1]:.1f}",lw=1)
    # ax[1,1].set_xlabel('Time')
    ax[2,1].set_ylabel('S21 Imag')

    ax[3,0].plot(lT,np.abs(lS21),'-')
    ax[3,0].set_xlabel('Time')
    ax[3,0].set_ylabel('Abs')

    ax[3,1].plot(lT,np.angle(lS21),'-')
    ax[3,1].set_xlabel('Time')
    ax[3,1].set_ylabel('Phase')
    
    for iax in ax.flatten():
        iax.grid()
        iax.legend()
    ax[0,0].set_title(f'F_lo = {lo_freq*1e-9:.3f} GHz')
    ax[0,1].set_title(f'F_lo = {lo_freq*1e-9:.3f} GHz')

    fig.tight_layout()

    fig2,ax2 = plt.subplots(figsize=(8,6))
    lf = np.arange(5.0e9,5.8e9,0.0001e9)
    lnotch = S21notch(nominal_par, lf)
    ax2.plot(lnotch.real,lnotch.imag,'-',color='gray')
    par_copy = copy.deepcopy(nominal_par)
    par_copy['Ql'] = Ql_T(Kinit)
    par_copy['Qc'] = Qc_T(Kinit)
    lnotch = S21notch(par_copy, lf)
    ax2.plot(lnotch.real,lnotch.imag,'--',color='gray')
    sc = ax2.scatter(lS21[::10].real,lS21[::10].imag,c=lT[::10],cmap='jet')
    fig2.colorbar(sc,ax=ax2,label='Time')

    ax2.grid()
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imag')
    ax2.set_title(f'F_lo = {lo_freq*1e-9:.3f} GHz')

    fig2.tight_layout()
    
    plt.show()
    


if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]))


