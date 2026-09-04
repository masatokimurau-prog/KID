import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import least_squares

import ds_style

def S21notch(par_, f_):
    env = par_['a']*np.exp(1j*par_['alpha'])*np.exp(-2j*np.pi*f_*par_['tau'])
    res = (par_['Ql']/np.abs(par_['Qc'])*np.exp(1j*par_['phi'])) / (1+2j*par_['Ql']*(f_/par_['fr']-1))
    return env*(1-res)


def funcPhase(f, tau, alpha):
    return np.exp(1j*(2*np.pi*f*tau + alpha))

def funcCircle(pars, x, y,):
    xc, yc, r = pars
    return (x-xc)**2 + (y-yc)**2 - r**2

def funcCircle_xaxis(pars, x, y,):
    xc, r = pars
    yc = 0
    return (x-xc)**2 + (y-yc)**2 - r**2

nominal_par = {'a':4.0, 'alpha':0.2, 'tau':20, 'Ql':450, 'Qc':475, 'phi':0, 'fr':5.5}

lf = np.linspace(5.450,5.550,101)
print(lf)
ls21 = S21notch(nominal_par, lf)

fr_idx = np.argmin(np.abs(ls21))
fr = lf[fr_idx]

a1,b1 = np.polyfit(lf[0:3],np.angle(ls21[0:3]),1)
a2,b2 = np.polyfit(lf[-6:-3],np.angle(ls21[-6:-3]),1)
tau = (a1+a2)/2/(2*np.pi)
alpha = np.angle(ls21[fr_idx]) - 2*np.pi*lf[fr_idx]*tau

ls21corr = ls21*np.exp(-1j*(2*np.pi*lf*tau + alpha))

circle_res = least_squares(funcCircle, x0=[np.max(np.abs(ls21corr)/2), 0, np.max(np.abs(ls21corr))], args=(np.real(ls21corr[fr_idx-10:fr_idx+10]),np.imag(ls21corr[fr_idx-10:fr_idx+10])))
norm0 = np.hypot(circle_res.x[0],circle_res.x[1])
norm1 = np.hypot(circle_res.x[0] + circle_res.x[2]*circle_res.x[0]/norm0, circle_res.x[1] + circle_res.x[2]*circle_res.x[1]/norm0)
norm2 = norm1-norm0

print(fr, tau, alpha)


fig,axs = plt.subplots(figsize=(18*0.75,9*0.75),ncols=2)
ax = axs[0]
ax.plot(ls21.real, ls21.imag, '-o', color='k', label='Observed Data')
ax.set_xlabel('ch0 [mV]')
ax.set_ylabel('ch1 [mV]')

ax = axs[1]
ax.plot(ls21corr.real/norm1, ls21corr.imag/norm1, '-o', color='k', label='Corrected Data')
ax.plot((circle_res.x[0] + circle_res.x[2]*np.cos(np.linspace(0,2*np.pi,100)))/norm1, (circle_res.x[1] + circle_res.x[2]*np.sin(np.linspace(0,2*np.pi,100)))/norm1,'--',color='gray',label='circle fit')
ax.plot(circle_res.x[0]/norm1,circle_res.x[1]/norm1,'*',ms=20,color='k',label='center')
ax.set_xlabel('I')
ax.set_ylabel('Q')

for ax in axs:
    ax.grid()
    ax.legend(facecolor='white', framealpha=0.75, frameon=True, loc='upper right', fontsize='medium')

# fig.tight_layout()
fig.subplots_adjust(wspace=0.3)

plt.show()
plt.close()


fig,axs = plt.subplots(figsize=(18*0.75,9*0.75),ncols=2)
ax = axs[0]
ax.plot(lf, np.abs(ls21), '--', color='gray', label='Observed data')
ax.plot(lf, np.abs(ls21corr), '-o', color='k', label='After correction')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Amplitude')
ax = axs[1]
ax.plot(lf, np.angle(ls21), '--', color='gray', label='Observed data')
ax.plot(lf, np.angle(ls21corr), '-o', color='k', label='After correction')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Phase')
ax.legend(facecolor='white', framealpha=0.75, frameon=True, loc='upper right', fontsize='medium')

for ax in axs:
    ax.grid()
    
# fig.tight_layout()
fig.subplots_adjust(wspace=0.3)

plt.show()



fig2,axs2 = plt.subplots(ncols=3)
ax = axs2[0]
ax.plot(ls21.real, ls21.imag, '-o', color='k')
ax = axs2[1]
ax.plot(lf, np.abs(ls21), '-o', color='k')
ax = axs2[2]
ax.plot(lf, np.angle(ls21), '-o', color='k')

plt.show()


