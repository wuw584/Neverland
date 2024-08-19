import numpy as np
import matplotlib.pyplot as plt
from DasTools import DasPrep as dp

import glob
from scipy import signal

import os

with open('das_params.txt', 'r') as f:
    das_params = f.readlines()


flist = glob.glob(os.path.join(das_params[0],'*.h5'))
flist.sort()

metadata = dp.read_das(flist[0], metadata=True)
  
dt = metadata['dt']
dx = metadata['dx']

data = np.concatenate([dp.read_das(fname) for fname in flist], axis=1)

M = 10
data = signal.decimate(data, M, axis=1)
dt *= M


# preprocessing / common-mode noise removal
data1 = dp.das_preprocess(data)
# data1 = dp.tapering(data1, alpha=0.2)
# data1 = dp.lowpass(data1, dt=metadata['dt'], fh=10)

Spg_all = []

for i in range(200, 2200):
    trace = data[i,:] / data[i,:].std()

    f1, t1, Spg = signal.spectrogram(trace, 1./dt, nperseg=300, noverlap=300*0.9)
    
    Spg_all.append(Spg)

Spg_all = np.array(Spg_all).mean(axis=0)

import matplotlib.gridspec as gridspec

# fig,ax = plt.subplots(ncols=1, nrows=5, ,figsize=[9,6], constrained_layout=True)

# fax1 = fig3.add_subplot(gs[0, :])

fig = plt.figure(figsize=[7,7], constrained_layout=True)
gs = fig.add_gridspec(5,1)

ax1 = fig.add_subplot(gs[:3,0])

data_plot = data1[:, :]

data_plot /= data_plot.std(axis=1, keepdims=True)
clim = data_plot.std() * 3
extent=[0, data_plot.shape[1]*dt, data_plot.shape[0]*dx/1000., 0]
ax1.imshow(data_plot[::10,:], aspect='auto', cmap='seismic', vmin=-clim, vmax=clim,
          extent=extent)
ax1.set_ylabel('Fiber distance (km)')

ax2 = fig.add_subplot(gs[3,0])
ax3 = fig.add_subplot(gs[4,0])

ax2.plot(np.arange(len(data1[1000,:]))*dt, 
         data1[1000,:] / abs(data1[1000,:]).max())
ax2.set_ylabel('Normalized Amplitude')
ax2.set_xlim([0, data_plot.shape[1]*dt])


clim = np.log10(Spg_all).max()
ax3.imshow(np.log10(Spg_all), aspect='auto', cmap='jet', vmin=-3, vmax=0.5, 
           origin='lower', extent=[t1[0], t1[-1], f1[0], f1[-1]])


ax3.set_ylim([0, 15])
ax3.set_xlim([0, data_plot.shape[1]*dt])
ax3.set_ylabel('Frequency (Hz)')
ax3.set_xlabel('Time (s)')

# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
plt.savefig("DAS_read.png")
plt.close()