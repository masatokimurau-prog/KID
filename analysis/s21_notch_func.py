import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import ds_style

def S21notch(par_, f_):
    env = par_['a']*np.exp(1j*par_['alpha'])*np.exp(-2j*np.pi*f_*par_['tau'])
    res = (par_['Ql']/np.abs(par_['Qc'])*np.exp(1j*par_['phi'])) / (1+2j*par_['Ql']*(f_/par_['fr']-1))
    return env*(1-res)

nominal_par = {'a':1.0, 'alpha':1.0, 'tau':-0.2, 'Ql':450, 'Qc':475, 'phi':0, 'fr':5.5}
lf = np.arange(3.0,6.0001,0.001)

leg_opt = {'columnspacing':1,'fontsize':'small','handlelength':1,'ncols':2}

pdf1 = PdfPages('pc1.pdf')

fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for ifr in np.arange(4.5,5.51,0.2):
    par = copy.deepcopy(nominal_par)
    par['fr'] = ifr

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{ifr:.2f} GHz')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{ifr:.2f} GHz')
    ax[2].plot(lf,np.angle(val),'-',label=f'{ifr:.2f} GHz')

def ax_label(ax,title):
    ax[0].set_xlabel('Re')
    ax[0].set_ylabel('Im')
    ax[1].set_xlabel('Freq [GHz]')
    ax[1].set_ylabel('Abs')
    ax[2].set_xlabel('Freq [GHz]')
    ax[2].set_ylabel('Arg')
    for iax in ax:
        iax.grid()
        iax.legend(**leg_opt,title=title)

ax_label(ax,'f_r')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for idia in np.arange(0.1,0.9,0.1):
    par = copy.deepcopy(nominal_par)
    par['Qc'] = par['Ql']/idia

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{idia:.2f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{idia:.2f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{idia:.2f}')

ax_label(ax,'d')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for iQl in np.arange(300,600,25):
    par = copy.deepcopy(nominal_par)
    par['Qc'] = iQl/((par['Ql']/par['Qc']))
    par['Ql'] = iQl

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{iQl:.0f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{iQl:.0f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{iQl:.0f}')

ax_label(ax,'Ql (fixed d)')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for ia in np.arange(0.0,2.0,0.2):
    par = copy.deepcopy(nominal_par)
    par['a'] = ia

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{ia:.1f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{ia:.1f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{ia:.1f}')

ax_label(ax,'a')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for ia in np.arange(-1.0,1.0,0.2):
    par = copy.deepcopy(nominal_par)
    par['alpha'] = ia

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{ia:.1f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{ia:.1f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{ia:.1f}')

ax_label(ax,'alpha')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for itau in np.arange(-0.5,0.5,0.25):
    par = copy.deepcopy(nominal_par)
    par['tau'] = itau

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{itau:.1f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{itau:.1f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{itau:.1f}')

ax_label(ax,'tau')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,4.5),ncols=3)
for iphi in np.arange(-np.pi,np.pi,np.pi/4):
    par = copy.deepcopy(nominal_par)
    par['phi'] = iphi

    val = S21notch(par, lf)

    ax[0].plot(val.real,val.imag,'-',label=f'{iphi:.2f}')
    ax[1].plot(lf,np.absolute(val),'-',label=f'{iphi:.2f}')
    ax[2].plot(lf,np.angle(val),'-',label=f'{iphi:.2f}')

ax_label(ax,'phi')
fig.tight_layout()
fig.savefig(pdf1,format='pdf')



pdf1.close()    
    
    

