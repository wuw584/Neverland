import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import os
from scipy import signal
from obspy import Stream, Trace

from obspy.io.segy.core import _read_segy

from nptdms import TdmsFile
from DasTools import DasPrep as dp

def get_das_file_time(das_filename):
    das_file_time_str = ' '.join(os.path.splitext(os.path.basename(das_filename))[0].split('_')[-2:])
    return np.datetime64(datetime.datetime.strptime(das_file_time_str, '%Y%m%d %H%M%S.%f'))

def read_das_data(filename, ch1, ch2):
    data = dp.read_das(filename, ch1=ch1, ch2=ch2)
    metadata = dp.read_das(filename, metadata=True)
    return data, metadata['dt'], metadata['nt']

# def read_das_data(filename, ch1, ch2):
#     st = _read_segy(filename)
#     data = np.asarray([st[i].data for i in range(ch1, ch2)])
#     return data, st[0].stats['delta'], st[0].stats['npts']

def extended_data_decimate(data_prev, data_curr, data_next, mlist, pad=0.2):
    
    M = mlist.prod()
    
    padding = int(pad*data_curr.shape[1]) // M * M
    data = np.concatenate([data_prev[:,-padding:], data_curr, data_next[:,:padding]], axis=1)
    
    if M>1:
        for m in mlist:
            data = signal.decimate(data, int(m), axis=1, zero_phase=True)

    t1 = padding // M
    t2 = t1 + data_curr.shape[1] // M 
#     print(t1, t2, padding, M, data.shape)
#     assert t2 + padding//M == data.shape[1]
    
    data = data[:,t1:t2]
    
    return data.astype('float32'), data_curr, data_next


def rolling_decimate(continous_data_list, ch1, ch2, mlist):
    data = np.empty((ch2-ch1, 0))
    data_prev, dt, *_ = read_das_data(continous_data_list[0], ch1, ch2)
    data_curr, *_ = read_das_data(continous_data_list[1], ch1, ch2)
    for file in continous_data_list[2:]:
        print(file)
        data_next, *_ = read_das_data(file, ch1, ch2)
        data_deci_new, data_prev, data_curr = extended_data_decimate(data_prev, data_curr, data_next, mlist)
        data = np.concatenate([data, data_deci_new], axis=1)
    mdt = dt * mlist.prod()
    return data, mdt


def das_processing(begin_date, interval, datafile, datafile_time, ch1, ch2, mlist):

    end_date = begin_date + interval 
    datafile_arg_choose = np.where((datafile_time>=begin_date)&(datafile_time<end_date))[0]
    datafile_arg_choose = np.r_[datafile_arg_choose[0]-1, datafile_arg_choose, datafile_arg_choose[-1]+1]
    datafile_choose = [datafile[i] for i in datafile_arg_choose]
    datafile_time_choose = datafile_time[datafile_arg_choose]
        
    data, dt = rolling_decimate(datafile_choose, ch1, ch2, mlist)
    return data, dt


def das_st_write_sac(das_tr, date_folder_path, write_coordinates=True):
    chn = int(das_tr.stats.location)
    if chn%3 == 0:
        das_tr.stats.channel = 'HHE'
    elif chn%3 == 1:
        das_tr.stats.channel = 'HHN'
    elif chn%3 == 2:
        das_tr.stats.channel = 'HHZ'
    if write_coordinates:
        das_tr.stats.sac = {'stla': das_tr.stats.coordinates['latitude'], 
                      'stlo': das_tr.stats.coordinates['longitude']}
    nw = das_tr.stats.network
    sta = das_tr.stats.station
    
    das_tr.write(date_folder_path + '\\'+nw+'.'+sta+'.'+str(chn//3*3)+'.'+das_tr.stats.channel+'.sac', format='SAC') 

