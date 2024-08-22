import numpy as np
import matplotlib.pyplot as plt
from DasTools import DasPrep as dp
from DasTools import GetParams as gp
import glob
from scipy import signal
import os
import matplotlib.gridspec as gridspec
import sys

##get params
param = gp.param(sys.argv[0])
data_dir = param["data_dir"]
das_read_clean_save_dir = param["das_read_clean_save_dir"]
das_ch1 = param["das_ch1"]
das_ch2 = param["das_ch2"]
M = param["M"]
norm_amp_ch = param["norm_amp_ch"]
plt_save_name = param["plt_save_name"]
# event_catalog_file = param["event_catalog_file"]
# event_cut_npy_save_dir = param["event_cut_npy_save_dir"]
# event_cut_plt_save_dir = param["event_cut_plot_save_dir" ]
# event_cut_file_name_prefix = param["event_cut_file_name_prefix"]
# dt_after = param["dt_after"]
# dt_before  =param["dt_before"]

# 创建文件夹
if not os.path.exists(das_read_clean_save_dir):
    os.makedirs(das_read_clean_save_dir)	# 创建文件夹

#das data
flist = glob.glob(os.path.join(data_dir,'*.h5'))
flist.sort()
# print(flist)
metadata = dp.read_das(flist[0], metadata=True)
dt = metadata['dt']
dx = metadata['dx']
nch = metadata['nch']
GL = metadata['GL']


#data read and downsample
data = np.concatenate([dp.read_das(fname) for fname in flist], axis=1)
data = data[das_ch1:das_ch2]
data = signal.decimate(data, M, axis=1)
dt *= M

# preprocessing / common-mode noise removal
data1 = dp.das_preprocess(data)
# data1 = dp.tapering(data1, alpha=0.2)
# data1 = dp.lowpass(data1, dt=metadata['dt'], fh=10)

#cal spectrogram
Spg_all = []

for i in range(0, len(data)):
    trace = data[i,:] / data[i,:].std()

    f1, t1, Spg = signal.spectrogram(trace, 1./dt, nperseg=300, noverlap=300*0.9)
    
    Spg_all.append(Spg)

Spg_all = np.array(Spg_all).mean(axis=0)


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

ax2.plot(np.arange(len(data1[norm_amp_ch,:]))*dt, 
         data1[norm_amp_ch,:] / abs(data1[norm_amp_ch,:]).max())
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
# if not os.path.exists(save_filename):
#     os.system(r"touch {}".format(save_filename)) #调用系统命令行来创建文件

plt.savefig( os.path.join(das_read_clean_save_dir, plt_save_name+'.png'))
plt.close()