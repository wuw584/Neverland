import numpy as np
import matplotlib.pyplot as plt
from DasTools import DasPrep as dp
from DasTools import GetParams as gp
from DasTools import DasCutEvent as dce
import glob
from scipy import signal
import os
import matplotlib.gridspec as gridspec
import sys

##get params
param = gp.param(sys.argv[0])
data_dir = param["data_dir"]
das_read_clean_save_dir = param["das_read_clean_save_dir"]
event_catalog_file = param["event_catalog_file"]
event_cut_npy_save_dir = param["event_cut_npy_save_dir"]
event_cut_plt_save_dir = param["event_cut_plot_save_dir" ]
event_cut_file_name_prefix = = ["event_cut_file_name_prefix"]

##catalog file
cat = pd.read_csv(hs_catalog_file, delim_whitespace=True)
cat_time = dce.get_cat_time(cat)
lat = hs_cat['Latitude'].values
lon = hs_cat['Longitude'].values

#plot event map
plt.figure(figsize=[7,7])
plt.plot(lon, lat,'.-')
plt.plot(lon[0], lat[0], 'r.')
plt.savefig(event_cut_plot_save_dir + "event_map.png")

#das data
flist = glob.glob(os.path.join(data_dir,'*.h5'))
flist.sort()
# print(flist , os.path.join(das_params[0],'*.h5'))
metadata = dp.read_das(flist[0], metadata=True)
dt = metadata['dt']
dx = metadata['dx']
nch = metadata['nch']
GL = metadata['GL']

#cut event and save
dt_before, dt_after = 1, 2
save_file_name_prefix =  event_cut_plt_save_dir + event_cut_file_name_prefix
dce.extract_das_data(das_file, cat_time, dt_before, dt_after, save_file_name_prefix, overwrite=True, verbose=True)

#read event data
ieq = 0
savename = save_file_name_prefix + str(ieq) + '.npy'
dt = 1./2000.
print(savename)
data = np.load(savename)

#data process
data1 = dp.das_preprocess(data)


#cal spectrogram
Spg_all = []
for i in range(0, nch):
    trace = data[i,:] / data[i,:].std()
    f1, t1, Spg = signal.spectrogram(trace, 1./dt, nperseg=300, noverlap=300*0.9)
    Spg_all.append(Spg)
Spg_all = np.array(Spg_all).mean(axis=0)


#plt
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
save_filename = das_params[1]
# if not os.path.exists(save_filename):
#     os.system(r"touch {}".format(save_filename)) #调用系统命令行来创建文件
if not os.path.exists(save_filename):
    os.makedirs(save_filename)	# 创建文件夹
plt.savefig( os.path.join(save_filename, 'Das_read.png'))
plt.close()