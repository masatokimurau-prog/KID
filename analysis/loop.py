import sys,os
import glob
import subprocess

lfile = glob.glob('April21st/wf_260421_2*.npz')
lfile = glob.glob('April21st/wf_260421_1[8-9]*.npz') + lfile
lfile = glob.glob('April21st/wf_260421_175*.npz') + lfile
lfile = glob.glob('April21st/wf_260422_0*.npz') + lfile
lfile = glob.glob('April21st/wf_260422_1[0-2]*.npz') + lfile
print(lfile)

for ifile in lfile:
    cmd = f'python plot_wf_Apr21.py {ifile}'
    subprocess.run(cmd,shell=True)

