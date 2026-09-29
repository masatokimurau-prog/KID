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

VERBOSE = 1

def load_group_from_csv(pattern, basedir):
    """Load the per-waveform parameter csv(s) matching `pattern` for one group.

    `pattern` is the same .npz glob pattern used in ana_forJPS.py's
    main_compare_distance(); for each matched <name>.npz we read the
    companion <name>_ana.csv written by that function (analyze_waveforms'
    dfres saved via dfres.to_csv(...)) instead of redoing the waveform
    analysis. All matched files are pooled into one DataFrame.

    Returns (dfgroup, sample_rate, tbin), or (None, None, None) if no
    companion csv was found for this pattern.
    """
    fns = sorted(glob.glob(os.path.join(basedir, pattern)))
    if len(fns) == 0:
        print(f'WARNING: no files matched pattern {pattern!r}')
        return None, None, None

    dfs = []
    for fn_ in fns:
        csv_fn = fn_[:-4] + '_ana.csv'
        if not os.path.exists(csv_fn):
            print(f'WARNING: {csv_fn!r} not found -- skipping')
            continue

        dfs.append(pd.read_csv(csv_fn))


    if len(dfs) == 0:
        return None, None, None

    dfgroup = pd.concat(dfs, ignore_index=True)
    return dfgroup


def main_laser_zscan(binwidth):
    patterns_labels = [ 
        ('Aug31st/wf_260831_15554?_*.npz', 4.00, 6.90),
        ('Aug31st/wf_260831_15562?_*.npz', 4.00, 6.70),
        ('Aug31st/wf_260831_15570?_*.npz', 4.00, 6.50),
        ('Aug31st/wf_260831_15575?_*.npz', 4.00, 6.30),
        ('Aug31st/wf_260831_15413?_*.npz', 4.00, 6.15),
        ('Aug31st/wf_260831_15421?_*.npz', 4.00, 6.00),
        ('Aug31st/wf_260831_15425?_*.npz', 4.00, 5.85),
        ('Aug31st/wf_260831_15434?_*.npz', 4.00, 5.70),
        ('Aug31st/wf_260831_16212?_*.npz', 4.00, 5.50),
        ('Aug31st/wf_260831_15474?_*.npz', 3.50, 6.90),
        ('Aug31st/wf_260831_15482?_*.npz', 3.50, 6.70),
        ('Aug31st/wf_260831_15490?_*.npz', 3.50, 6.50),
        ('Aug31st/wf_260831_15494?_*.npz', 3.50, 6.30),
        ('Aug31st/wf_260831_15502?_*.npz', 3.50, 6.00),
        ('Aug31st/wf_260831_15514?_*.npz', 3.50, 5.70),
        ('Aug31st/wf_260831_16203?_*.npz', 3.50, 5.50),
        ]

    basedir = os.path.dirname(os.path.abspath(__file__))

    dgroups = {}
    for pattern, labelx, labelz in patterns_labels:
        dfgroup = load_group_from_csv(pattern, basedir)
        if dfgroup is None:
            print(f'WARNING: nothing to load for pattern {pattern!r} -- skipping "{labelx} and {labelz}"')
            continue

        dgroups[(labelx,labelz)] = dfgroup

    # cut = f'proj_integ > 1.5 and proj_integ_left > 0 and proj_integ_right > 0 and proj_e_max_left > 0 and proj_e_max_right > 0'
    cut = 'proj_integ_left > 0 and proj_integ_right > 0 and proj_e_max_10ns_left > 0 and proj_e_max_10ns_right > 0'

    result = []
    for (labelx,labelz), dfgroup in dgroups.items():
        dfcut = dfgroup.query(cut)

        dres = {}
        dres['x'] = labelx
        dres['z'] = labelz

        dres['all_proj_integ_mean'] = np.mean(dfgroup['proj_integ_left']+dfgroup['proj_integ_right'])
        dres['all_proj_integ_err'] = np.std(dfgroup['proj_integ_left']+dfgroup['proj_integ_right'])/np.sqrt(len(dfgroup))
        dres['all_proj_peak_mean'] = dfgroup['proj_max_10ns'].mean()*1e3
        dres['all_proj_peak_err'] = dfgroup['proj_max_10ns'].std()*1e3/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_left_mean'] = dfgroup['proj_e_max_10ns_left'].mean()
        dres['all_proj_e_max_left_err'] = dfgroup['proj_e_max_10ns_left'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_right_mean'] = dfgroup['proj_e_max_10ns_right'].mean()
        dres['all_proj_e_max_right_err'] = dfgroup['proj_e_max_10ns_right'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_tau_left_mean'] = np.mean(dfgroup['proj_integ_left']/dfgroup['proj_max_10ns'])*binwidth
        dres['all_proj_tau_left_err'] = np.std(dfgroup['proj_integ_left']/dfgroup['proj_max_10ns'])*binwidth
        dres['all_proj_tau_right_mean'] = np.mean(dfgroup['proj_integ_right']/dfgroup['proj_max_10ns'])*binwidth
        dres['all_proj_tau_right_err'] = np.std(dfgroup['proj_integ_right']/dfgroup['proj_max_10ns'])*binwidth

        dres['proj_integ_mean'] = dfcut['proj_integ'].mean()
        dres['proj_integ_err'] = dfcut['proj_integ'].std()/np.sqrt(len(dfcut))
        dres['proj_peak_mean'] = dfcut['proj_max_10ns'].mean()*1e3
        dres['proj_peak_err'] = dfcut['proj_max_10ns'].std()*1e3/np.sqrt(len(dfcut))
        dres['proj_e_max_left_mean'] = dfcut['proj_e_max_10ns_left'].mean()
        dres['proj_e_max_left_err'] = dfcut['proj_e_max_10ns_left'].std()/np.sqrt(len(dfcut))
        dres['proj_e_max_right_mean'] = dfcut['proj_e_max_10ns_right'].mean()
        dres['proj_e_max_right_err'] = dfcut['proj_e_max_10ns_right'].std()/np.sqrt(len(dfcut))
        dres['proj_tau_left_mean'] = np.mean(dfcut['proj_integ_left']/dfcut['proj_max_10ns'])*binwidth
        dres['proj_tau_left_err'] = np.std(dfcut['proj_integ_left']/dfcut['proj_max_10ns'])*binwidth
        dres['proj_tau_right_mean'] = np.mean(dfcut['proj_integ_right']/dfcut['proj_max_10ns'])*binwidth
        dres['proj_tau_right_err'] = np.std(dfcut['proj_integ_right']/dfcut['proj_max_10ns'])*binwidth

        result.append(dres)

    dfres = pd.DataFrame(result)
    print(dfres)

    lz = dfres['z'].unique()
    zcol = {z: mpl.cm.jet(idx/len(lz)) for idx, z in enumerate(sorted(lz))}

    fig,axs = plt.subplots(figsize=(10,10), nrows=3, ncols=2, sharex=True)
    plot_all = False
    for idx,icut in enumerate(['x==4.0','x==3.5']):
        ax = axs[0,0]
        df = dfres.query(icut)
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_integ_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_integ_mean'], yerr=df['proj_integ_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[0,1]
        df = dfres.query(icut)
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_peak_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_peak_mean'], yerr=df['proj_peak_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[1,0]
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_tau_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_tau_left_mean'], yerr=df['proj_tau_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)
        ax = axs[1,1]
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_e_max_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_e_max_left_mean'], yerr=df['proj_e_max_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[2,0]
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_tau_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_tau_right_mean'], yerr=df['proj_tau_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)
        ax = axs[2,1]
        if plot_all:
            ax.errorbar(df['z'], df['all_proj_e_max_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['z'], df['proj_e_max_right_mean'], yerr=df['proj_e_max_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

    axs[0,0].set_ylabel('proj integ [arb]')
    axs[0,1].set_ylabel('proj peak [mV]')
    axs[1,0].set_ylabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
    axs[2,0].set_ylabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
    axs[1,1].set_ylabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
    axs[2,1].set_ylabel(r'$\tau$$_{d}$ (height) [$\mu$s]')
    for idx in range(2):
        axs[1,idx].set_ylim(0, 0.25)
        axs[2,idx].set_ylim(0, 1.2)
        axs[2,idx].set_xlabel('z [mm]')

    for ax in axs.flatten():
        ax.grid()
        ax.legend()

    fig.tight_layout()
    fig.savefig('pc1.png')

    fig2,axs2 = plt.subplots(figsize=(10,10), nrows=3, ncols=2)
    fig3,axs3 = plt.subplots(figsize=(10,10), nrows=3, ncols=2)
    binped = np.linspace(-25, 25, 101)
    bintmax = np.arange(0, 0.2, 0.002)
    bintaur = np.arange(0, 0.251, 0.002)
    bintaud = np.arange(0, 1.21, 0.01)
    binq = np.linspace(0, 12.1, 101)
    binv = np.linspace(0, 40.1, 101)

    for idx, ((labelx,labelz), dfgroup) in enumerate(dgroups.items()):
        if labelx == 4.0: ax = axs2
        elif labelx == 3.5: ax = axs3

        df = dfgroup.query(cut)

        iax = ax[0,0]
        hy,_ = np.histogram(df['proj_integ_left']+df['proj_integ_right'], bins=binq)
        iax.hist(binq[:-1], binq, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')

        iax = ax[1,0]
        hy,_ = np.histogram(df['proj_integ_left']/df['proj_max_10ns']*binwidth, bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')

        iax = ax[2,0]
        hy,_ = np.histogram(df['proj_integ_right']/df['proj_max_10ns']*binwidth, bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')

        iax = ax[0,1]
        hy,_ = np.histogram(df['proj_max_10ns']*1e3, bins=binv)
        iax.hist(binv[:-1], binv, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')

        iax = ax[1,1]
        hy,_ = np.histogram(df['proj_e_max_10ns_left'], bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')  

        iax = ax[2,1]
        hy,_ = np.histogram(df['proj_e_max_10ns_right'], bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=zcol[labelz], label=f'{labelz:.2f} mm')

    for ax in [axs2,axs3]:
        for iax in ax.flatten():
            iax.grid()
            iax.legend(fontsize='xx-small',ncol=2)
        ax[0,0].set_xlabel('Integral')
        ax[1,0].set_xlabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
        ax[2,0].set_xlabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
        ax[0,1].set_xlabel('proj Peak [mV]')
        ax[1,1].set_xlabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
        ax[2,1].set_xlabel(r'$\tau$$_{d}$ (height) [$\mu$s]')

    for iax in axs2.flatten():
        iax.set_title('x = 4.0 mm')
    for iax in axs3.flatten():
        iax.set_title('x = 3.5 mm') 

    fig2.tight_layout()
    fig3.tight_layout()
    fig2.savefig('pc2.png')
    fig3.savefig('pc3.png')


    plt.show()



def main_laser_xscan(binwidth):
    patterns_labels = [ 
        ('Aug25th/wf_260825_13244?_*.npz', 4.30, 6.30),
        ('Aug25th/wf_260825_13192?_*.npz', 4.50, 6.30),
        ('Aug25th/wf_260825_13151?_*.npz', 4.60, 6.30),
        ('Aug25th/wf_260825_13052?_*.npz', 4.75, 6.30),
        ('Aug25th/wf_260825_13015?_*.npz', 4.95, 6.30),
        ('Aug25th/wf_260825_13421?_*.npz', 5.10, 6.30),
        ('Aug25th/wf_260825_13371?_*.npz', 5.25, 6.30),
        ('Aug25th/wf_260825_13272?_*.npz', 5.60, 6.30),
        ('Aug25th/wf_260825_13240?_*.npz', 4.30, 6.00),
        ('Aug25th/wf_260825_13200?_*.npz', 4.50, 6.00),
        ('Aug25th/wf_260825_13125?_*.npz', 4.60, 6.00),
        ('Aug25th/wf_260825_13061?_*.npz', 4.75, 6.00),
        ('Aug25th/wf_260825_13562?_*.npz', 4.95, 6.00),
        ('Aug25th/wf_260825_13444?_*.npz', 5.10, 6.00),
        ('Aug25th/wf_260825_13350?_*.npz', 5.25, 6.00),
        ('Aug25th/wf_260825_13281?_*.npz', 5.60, 6.00),
        ]

    basedir = os.path.dirname(os.path.abspath(__file__))

    dgroups = {}
    for pattern, labelx, labelz in patterns_labels:
        dfgroup = load_group_from_csv(pattern, basedir)
        if dfgroup is None:
            print(f'WARNING: nothing to load for pattern {pattern!r} -- skipping "{labelx} and {labelz}"')
            continue

        dgroups[(labelx,labelz)] = dfgroup

    # cut = f'proj_integ > 1.5 and proj_integ_left > 0 and proj_integ_right > 0 and proj_e_max_left > 0 and proj_e_max_right > 0'
    cut = 'proj_integ_left > 0 and proj_integ_right > 0 and proj_e_max_10ns_left > 0 and proj_e_max_10ns_right > 0'

    result = []
    for (labelx,labelz), dfgroup in dgroups.items():
        dfcut = dfgroup.query(cut)

        dres = {}
        dres['x'] = labelx
        dres['z'] = labelz

        # dres['all_proj_integ_mean'] = np.mean(dfgroup['proj_integ_left']+dfgroup['proj_integ_right'])
        # dres['all_proj_integ_err'] = np.std(dfgroup['proj_integ_left']+dfgroup['proj_integ_right'])/np.sqrt(len(dfgroup))
        # dres['all_proj_peak_mean'] = dfgroup['proj_max_10ns'].mean()*1e3
        # dres['all_proj_peak_err'] = dfgroup['proj_max_10ns'].std()*1e3/np.sqrt(len(dfgroup))
        # dres['all_proj_e_max_left_mean'] = dfgroup['proj_e_max_10ns_left'].mean()
        # dres['all_proj_e_max_left_err'] = dfgroup['proj_e_max_10ns_left'].std()/np.sqrt(len(dfgroup))
        # dres['all_proj_e_max_right_mean'] = dfgroup['proj_e_max_10ns_right'].mean()
        # dres['all_proj_e_max_right_err'] = dfgroup['proj_e_max_10ns_right'].std()/np.sqrt(len(dfgroup))
        # dres['all_proj_tau_left_mean'] = np.mean(dfgroup['proj_integ_left']/dfgroup['proj_max_10ns'])*binwidth
        # dres['all_proj_tau_left_err'] = np.std(dfgroup['proj_integ_left']/dfgroup['proj_max_10ns'])*binwidth
        # dres['all_proj_tau_right_mean'] = np.mean(dfgroup['proj_integ_right']/dfgroup['proj_max_10ns'])*binwidth
        # dres['all_proj_tau_right_err'] = np.std(dfgroup['proj_integ_right']/dfgroup['proj_max_10ns'])*binwidth

        # dres['proj_integ_mean'] = np.mean(dfcut['proj_integ_left']+dfcut['proj_integ_right'])
        # dres['proj_integ_err'] = np.std(dfcut['proj_integ_left']+dfcut['proj_integ_right'])/np.sqrt(len(dfcut))
        # dres['proj_peak_mean'] = dfcut['proj_max_10ns'].mean()*1e3
        # dres['proj_peak_err'] = dfcut['proj_max_10ns'].std()*1e3/np.sqrt(len(dfcut))
        # dres['proj_e_max_left_mean'] = dfcut['proj_e_max_10ns_left'].mean()
        # dres['proj_e_max_left_err'] = dfcut['proj_e_max_10ns_left'].std()/np.sqrt(len(dfcut))
        # dres['proj_e_max_right_mean'] = dfcut['proj_e_max_10ns_right'].mean()
        # dres['proj_e_max_right_err'] = dfcut['proj_e_max_10ns_right'].std()/np.sqrt(len(dfcut))
        # dres['proj_tau_left_mean'] = np.mean(dfcut['proj_integ_left']/dfcut['proj_max_10ns'])*binwidth
        # dres['proj_tau_left_err'] = np.std(dfcut['proj_integ_left']/dfcut['proj_max_10ns'])*binwidth
        # dres['proj_tau_right_mean'] = np.mean(dfcut['proj_integ_right']/dfcut['proj_max_10ns'])*binwidth
        # dres['proj_tau_right_err'] = np.std(dfcut['proj_integ_right']/dfcut['proj_max_10ns'])*binwidth

        dres['all_proj_integ_mean'] = np.mean(dfgroup['ch0_integ_left']+dfgroup['ch0_integ_right'])
        dres['all_proj_integ_err'] = np.std(dfgroup['ch0_integ_left']+dfgroup['ch0_integ_right'])/np.sqrt(len(dfgroup))
        dres['all_proj_peak_mean'] = dfgroup['ch0_peak_max_10ns'].mean()*1e3
        dres['all_proj_peak_err'] = dfgroup['ch0_peak_max_10ns'].std()*1e3/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_left_mean'] = dfgroup['ch0_e_max_10ns_left'].mean()
        dres['all_proj_e_max_left_err'] = dfgroup['ch0_e_max_10ns_left'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_right_mean'] = dfgroup['ch0_e_max_10ns_right'].mean()
        dres['all_proj_e_max_right_err'] = dfgroup['ch0_e_max_10ns_right'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_tau_left_mean'] = np.mean(dfgroup['ch0_integ_left']/dfgroup['ch0_peak_max_10ns'])*binwidth
        dres['all_proj_tau_left_err'] = np.std(dfgroup['ch0_integ_left']/dfgroup['ch0_peak_max_10ns'])*binwidth
        dres['all_proj_tau_right_mean'] = np.mean(dfgroup['ch0_integ_right']/dfgroup['ch0_peak_max_10ns'])*binwidth
        dres['all_proj_tau_right_err'] = np.std(dfgroup['ch0_integ_right']/dfgroup['ch0_peak_max_10ns'])*binwidth

        dres['proj_integ_mean'] = np.mean(dfcut['ch0_integ_left']+dfcut['ch0_integ_right'])
        dres['proj_integ_err'] = np.std(dfcut['ch0_integ_left']+dfcut['ch0_integ_right'])/np.sqrt(len(dfcut))
        dres['proj_peak_mean'] = dfcut['ch0_peak_max_10ns'].mean()*1e3
        dres['proj_peak_err'] = dfcut['ch0_peak_max_10ns'].std()*1e3/np.sqrt(len(dfcut))
        dres['proj_e_max_left_mean'] = dfcut['ch0_e_max_10ns_left'].mean()
        dres['proj_e_max_left_err'] = dfcut['ch0_e_max_10ns_left'].std()/np.sqrt(len(dfcut))
        dres['proj_e_max_right_mean'] = dfcut['ch0_e_max_10ns_right'].mean()
        dres['proj_e_max_right_err'] = dfcut['ch0_e_max_10ns_right'].std()/np.sqrt(len(dfcut))
        dres['proj_tau_left_mean'] = np.mean(dfcut['ch0_integ_left']/dfcut['ch0_peak_max_10ns'])*binwidth
        dres['proj_tau_left_err'] = np.std(dfcut['ch0_integ_left']/dfcut['ch0_peak_max_10ns'])*binwidth
        dres['proj_tau_right_mean'] = np.mean(dfcut['ch0_integ_right']/dfcut['ch0_peak_max_10ns'])*binwidth
        dres['proj_tau_right_err'] = np.std(dfcut['ch0_integ_right']/dfcut['ch0_peak_max_10ns'])*binwidth

        result.append(dres)

    dfres = pd.DataFrame(result)
    print(dfres)

    lx = dfres['x'].unique()
    xcol = {x: mpl.cm.jet(idx/len(lx)) for idx, x in enumerate(sorted(lx))}
    
    fig,axs = plt.subplots(figsize=(10,10), nrows=3, ncols=2, sharex=True)
    plot_all = False
    for idx,icut in enumerate(['z==6.3','z==6.0']):
        ax = axs[0,0]
        df = dfres.query(icut)
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_integ_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_integ_mean'], yerr=df['proj_integ_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[0,1]
        df = dfres.query(icut)
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_peak_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_peak_mean'], yerr=df['proj_peak_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[1,0]
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_tau_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_tau_left_mean'], yerr=df['proj_tau_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)
        ax = axs[1,1]
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_e_max_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_e_max_left_mean'], yerr=df['proj_e_max_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

        ax = axs[2,0]
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_tau_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_tau_right_mean'], yerr=df['proj_tau_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)
        ax = axs[2,1]
        if plot_all:
            ax.errorbar(df['x'], df['all_proj_e_max_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['x'], df['proj_e_max_right_mean'], yerr=df['proj_e_max_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=icut)

    axs[0,0].set_ylabel('proj integ [arb]')
    axs[0,1].set_ylabel('proj peak [mV]')
    axs[1,0].set_ylabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
    axs[2,0].set_ylabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
    axs[1,1].set_ylabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
    axs[2,1].set_ylabel(r'$\tau$$_{d}$ (height) [$\mu$s]')
    for idx in range(2):
        axs[1,idx].set_ylim(0, 0.25)
        axs[2,idx].set_ylim(0, 1.2)
        axs[2,idx].set_xlabel('x [mm]')

    for ax in axs.flatten():
        ax.grid()
        ax.legend()

    fig.tight_layout()
    fig.savefig('pc1.png')

    fig2,axs2 = plt.subplots(figsize=(10,10), nrows=3, ncols=2)
    fig3,axs3 = plt.subplots(figsize=(10,10), nrows=3, ncols=2)
    binped = np.linspace(-25, 25, 101)
    bintmax = np.arange(0, 0.2, 0.002)
    bintaur = np.arange(0, 0.4001, 0.002)
    bintaud = np.arange(0, 1.51, 0.01)
    binq = np.linspace(0, 20.1, 101)
    binv = np.linspace(0, 40.1, 101)

    for idx, ((labelx,labelz), dfgroup) in enumerate(dgroups.items()):
        if labelz == 6.3: ax = axs2
        elif labelz == 6.0: ax = axs3

        df = dfgroup.query(cut)

        iax = ax[0,0]
        hy,_ = np.histogram(df['proj_integ_left']+df['proj_integ_right'], bins=binq)
        iax.hist(binq[:-1], binq, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')

        iax = ax[1,0]
        hy,_ = np.histogram(df['proj_integ_left']/df['proj_max_10ns']*binwidth, bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')

        iax = ax[2,0]
        hy,_ = np.histogram(df['proj_integ_right']/df['proj_max_10ns']*binwidth, bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')

        iax = ax[0,1]
        hy,_ = np.histogram(df['proj_max_10ns']*1e3, bins=binv)
        iax.hist(binv[:-1], binv, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')

        iax = ax[1,1]
        hy,_ = np.histogram(df['proj_e_max_10ns_left'], bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')  

        iax = ax[2,1]
        hy,_ = np.histogram(df['proj_e_max_10ns_right'], bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=xcol[labelx], label=f'{labelx:.2f} mm')

    for ax in [axs2,axs3]:
        for iax in ax.flatten():
            iax.grid()
            iax.legend(fontsize='xx-small',ncol=2)
        ax[0,0].set_xlabel('Integral')
        ax[1,0].set_xlabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
        ax[2,0].set_xlabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
        ax[0,1].set_xlabel('proj Peak [mV]')
        ax[1,1].set_xlabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
        ax[2,1].set_xlabel(r'$\tau$$_{d}$ (height) [$\mu$s]')

    for iax in axs2.flatten():
        iax.set_title('z = 6.3 mm')
    for iax in axs3.flatten():
        iax.set_title('z = 6.0 mm') 

    fig2.tight_layout()
    fig3.tight_layout()
    fig2.savefig('pc2.png')
    fig3.savefig('pc3.png')


    plt.show()



def main_laser_tscan(binwidth):
    # three datasets, one per (x,z) position, each scanning temperature
    dataset_patterns = [
        ('(4.00, 6.30)', [  # Temp scan at (x,z) = (4.00, 6.30) mm
            ('Sep1st/wf_260901_14284?_*.npz', 3.90),
            ('Sep1st/wf_260901_15450?_*.npz', 4.20),
            ('Sep1st/wf_260901_16103?_*.npz', 4.55),
            ('Sep1st/wf_260901_16461?_*.npz', 4.90),
            ('Sep1st/wf_260901_17042?_*.npz', 5.30),
            ('Sep1st/wf_260901_17300?_*.npz', 5.95),
        ]),
        ('(4.00, 6.00)', [  # Temp scan at (x,z) = (4.00, 6.00) mm
            ('Sep1st/wf_260901_14232?_*.npz', 3.90),
            ('Sep1st/wf_260901_15462?_*.npz', 4.20),
            ('Sep1st/wf_260901_16082?_*.npz', 4.55),
            ('Sep1st/wf_260901_16445?_*.npz', 4.90),
            ('Sep1st/wf_260901_17055?_*.npz', 5.30),
            ('Sep1st/wf_260901_17284?_*.npz', 5.95),
        ]),
        ('(3.50, 6.00)', [  # Temp scan at (x,z) = (3.50, 6.00) mm
            ('Sep1st/wf_260901_14560?_*.npz', 3.90),
            ('Sep1st/wf_260901_15351?_*.npz', 4.20),
            ('Sep1st/wf_260901_16164?_*.npz', 4.55),
            ('Sep1st/wf_260901_16431?_*.npz', 4.90),
            ('Sep1st/wf_260901_17072?_*.npz', 5.30),
            ('Sep1st/wf_260901_17271?_*.npz', 5.95),
        ]),
    ]
    patterns_labels = [(pattern, pos_label, labelT)
                        for pos_label, plist in dataset_patterns
                        for pattern, labelT in plist]

    basedir = os.path.dirname(os.path.abspath(__file__))

    dgroups = {}
    for pattern, pos_label, labelT in patterns_labels:
        dfgroup = load_group_from_csv(pattern, basedir)
        if dfgroup is None:
            print(f'WARNING: nothing to load for pattern {pattern!r} -- skipping "{pos_label} at {labelT} K"')
            continue

        dgroups[(pos_label, labelT)] = dfgroup

    cut = f'proj_integ > 1.5 and proj_integ_left > 0 and proj_integ_right > 0 and proj_e_max_left > 0 and proj_e_max_right > 0'

    result = []
    for (pos_label, labelT), dfgroup in dgroups.items():
        dfcut = dfgroup.query(cut)

        dres = {}
        dres['pos'] = pos_label
        dres['T'] = labelT

        dres['all_proj_integ_mean'] = dfgroup['proj_integ'].mean()
        dres['all_proj_integ_err'] = dfgroup['proj_integ'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_peak_mean'] = dfgroup['proj_max'].mean()*1e3
        dres['all_proj_peak_err'] = dfgroup['proj_max'].std()*1e3/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_left_mean'] = dfgroup['proj_e_max_left'].mean()
        dres['all_proj_e_max_left_err'] = dfgroup['proj_e_max_left'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_e_max_right_mean'] = dfgroup['proj_e_max_right'].mean()
        dres['all_proj_e_max_right_err'] = dfgroup['proj_e_max_right'].std()/np.sqrt(len(dfgroup))
        dres['all_proj_tau_left_mean'] = np.mean(dfgroup['proj_integ_left']/dfgroup['proj_max'])*binwidth
        dres['all_proj_tau_left_err'] = np.std(dfgroup['proj_integ_left']/dfgroup['proj_max'])*binwidth
        dres['all_proj_tau_right_mean'] = np.mean(dfgroup['proj_integ_right']/dfgroup['proj_max'])*binwidth
        dres['all_proj_tau_right_err'] = np.std(dfgroup['proj_integ_right']/dfgroup['proj_max'])*binwidth

        dres['proj_integ_mean'] = dfcut['proj_integ'].mean()
        dres['proj_integ_err'] = dfcut['proj_integ'].std()/np.sqrt(len(dfcut))
        dres['proj_peak_mean'] = dfcut['proj_max'].mean()*1e3
        dres['proj_peak_err'] = dfcut['proj_max'].std()*1e3/np.sqrt(len(dfcut))
        dres['proj_e_max_left_mean'] = dfcut['proj_e_max_left'].mean()
        dres['proj_e_max_left_err'] = dfcut['proj_e_max_left'].std()/np.sqrt(len(dfcut))
        dres['proj_e_max_right_mean'] = dfcut['proj_e_max_right'].mean()
        dres['proj_e_max_right_err'] = dfcut['proj_e_max_right'].std()/np.sqrt(len(dfcut))
        dres['proj_tau_left_mean'] = np.mean(dfcut['proj_integ_left']/dfcut['proj_max'])*binwidth
        dres['proj_tau_left_err'] = np.std(dfcut['proj_integ_left']/dfcut['proj_max'])*binwidth
        dres['proj_tau_right_mean'] = np.mean(dfcut['proj_integ_right']/dfcut['proj_max'])*binwidth
        dres['proj_tau_right_err'] = np.std(dfcut['proj_integ_right']/dfcut['proj_max'])*binwidth

        result.append(dres)

    dfres = pd.DataFrame(result)
    print(dfres)

    lt = dfres['T'].unique()
    tcol = {T: mpl.cm.jet(idx/len(lt)) for idx, T in enumerate(sorted(lt))}
    positions = [pos_label for pos_label, _ in dataset_patterns]

    fig,axs = plt.subplots(figsize=(10,10), nrows=3, ncols=2, sharex=True)
    plot_all = False
    for idx, ipos in enumerate(positions):
        ax = axs[0,0]
        df = dfres[dfres['pos'] == ipos].sort_values('T')
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_integ_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_integ_mean'], yerr=df['proj_integ_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)

        ax = axs[0,1]
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_peak_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_peak_mean'], yerr=df['proj_peak_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)

        ax = axs[1,0]
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_tau_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_tau_left_mean'], yerr=df['proj_tau_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)
        ax = axs[1,1]
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_e_max_left_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_e_max_left_mean'], yerr=df['proj_e_max_left_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)

        ax = axs[2,0]
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_tau_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_tau_right_mean'], yerr=df['proj_tau_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)
        ax = axs[2,1]
        if plot_all:
            ax.errorbar(df['T'], df['all_proj_e_max_right_mean'], fmt='--', color=f'C{idx}')
        ax.errorbar(df['T'], df['proj_e_max_right_mean'], yerr=df['proj_e_max_right_err'], fmt='-o', capsize=5, color=f'C{idx}', label=ipos)

    axs[0,0].set_ylabel('proj integ [arb]')
    axs[0,1].set_ylabel('proj peak [mV]')
    axs[1,0].set_ylabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
    axs[2,0].set_ylabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
    axs[1,1].set_ylabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
    axs[2,1].set_ylabel(r'$\tau$$_{d}$ (height) [$\mu$s]')
    for idx in range(2):
        axs[1,idx].set_ylim(0, 0.25)
        axs[2,idx].set_ylim(0, 1.2)
        axs[2,idx].set_xlabel('T [K]')

    axs[0,0].set_ylim(0,20)
    axs[0,1].set_ylim(0,20)
    for ax in axs.flatten():
        ax.grid()
        ax.legend()
    
    fig.tight_layout()
    fig.savefig('pc1_tscan.png')

    binped = np.linspace(-25, 25, 101)
    bintmax = np.arange(0, 0.2, 0.002)
    bintaur = np.arange(0, 0.51, 0.002)
    bintaud = np.arange(0, 1.51, 0.01)
    binq = np.linspace(0, 16.1, 101)
    binv = np.linspace(0, 40.1, 101)

    figs = {ipos: plt.subplots(figsize=(10,10), nrows=3, ncols=2) for ipos in positions}

    for (pos_label, labelT), dfgroup in dgroups.items():
        fig_pos, ax = figs[pos_label]

        df = dfgroup.query(cut)

        iax = ax[0,0]
        hy,_ = np.histogram(df['proj_integ'], bins=binq)
        iax.hist(binq[:-1], binq, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

        iax = ax[1,0]
        hy,_ = np.histogram(df['proj_integ_left']/df['proj_max']*binwidth, bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

        iax = ax[2,0]
        hy,_ = np.histogram(df['proj_integ_right']/df['proj_max']*binwidth, bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

        iax = ax[0,1]
        hy,_ = np.histogram(df['proj_max']*1e3, bins=binv)
        iax.hist(binv[:-1], binv, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

        iax = ax[1,1]
        hy,_ = np.histogram(df['proj_e_max_left'], bins=bintaur)
        iax.hist(bintaur[:-1], bintaur, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

        iax = ax[2,1]
        hy,_ = np.histogram(df['proj_e_max_right'], bins=bintaud)
        iax.hist(bintaud[:-1], bintaud, weights=hy, histtype='step', color=tcol[labelT], label=f'{labelT:.2f} K')

    for ipos, (fig_pos, ax) in figs.items():
        for iax in ax.flatten():
            iax.grid()
            iax.legend(fontsize='xx-small', ncol=2)
        ax[0,0].set_xlabel('Integral')
        ax[1,0].set_xlabel(r'$\tau$$_{r}$ (integ) [$\mu$s]')
        ax[2,0].set_xlabel(r'$\tau$$_{d}$ (integ) [$\mu$s]')
        ax[0,1].set_xlabel('proj Peak [mV]')
        ax[1,1].set_xlabel(r'$\tau$$_{r}$ (height) [$\mu$s]')
        ax[2,1].set_xlabel(r'$\tau$$_{d}$ (height) [$\mu$s]')
        for iax in ax.flatten():
            iax.set_title(f'(x,z) = {ipos} mm')
        fig_pos.tight_layout()

    for idx, ipos in enumerate(positions):
        figs[ipos][0].savefig(f'pc{2+idx}_tscan.png')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == 'Laser_zscan':
        main_laser_zscan(float(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == 'Laser_xscan':
        main_laser_xscan(float(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'Laser_tscan':
        main_laser_tscan(float(sys.argv[2]))