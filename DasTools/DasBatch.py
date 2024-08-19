import h5py 
import glob
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from DasTools import DasPrep as dp
import os
import datetime
import matplotlib
import logging
import csv
import time
import pandas as pd
# from obspy import Stream, Trace
# from obspy.io.segy.core import _read_segy
from nptdms import TdmsFile
import gc
from multiprocessing import Pool
from functools import partial
##############################################################################################


##############################################################################################


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


##############################################################################################
def get_continuous_segments(fname_datetime, file_len, tol):
    fname_datetime_diff = np.diff(fname_datetime)
    file_len_timedelta = datetime.timedelta(seconds=file_len*1.0001)
    segment_diff = np.where(fname_datetime_diff > file_len_timedelta)[0]  # define continuous segments by no files seperated more than the file length (15s)
    segment_start = np.r_[0, segment_diff + 1]
    segment_end = np.r_[segment_diff, len(fname_datetime) - 1]

    segment_start_datetime = fname_datetime[segment_start] + file_len_timedelta * 1.5 # shift to later by 1.5 file length to give buffer for rolling 
    segment_end_datetime = fname_datetime[segment_end] - file_len_timedelta * 1.5  # shift to earlier by 1.5 file length to give buffer for rolling 

    continuous_segment_size = np.array([x.total_seconds() for x in (segment_end_datetime - segment_start_datetime)])
    
    segment_choose = np.where(continuous_segment_size > tol)[0] 
    return segment_start_datetime[segment_choose], segment_end_datetime[segment_choose], continuous_segment_size[segment_choose]



def overview(datapath):
    datapath =  'L:\\anyuan_mine_2\\'
    datafile = glob.glob(datapath+'*.h5')
    datafile.sort()
    print("--------------overview of dir-------------------------")

    fname_format = datapath + 'anyuan_GL_5m_frq_1kHz_sp_1m_UTC_%Y%m%d_%H%M%S.%f.h5'
    fname_npdatetime = np.array([np.datetime64(datetime.datetime.strptime(x, fname_format),'us') for x in datafile])
    fname_datetime = np.array([datetime.datetime.strptime(x, fname_format) for x in datafile])


    metadata = dp.read_das(datafile[len(datafile)//2], metadata=True)

    for key in metadata.keys():
        print(key, ':', metadata[key])
        

    dt = metadata['dt']
    nt = metadata['nt']
    file_len = dt*nt



    print(dt,nt, file_len)

    
    segment_start_datetime, segment_end_datetime, continuous_segment_size = get_continuous_segments(fname_datetime, file_len, tol=10*60) # segments lasting more than 20 min
    print(continuous_segment_size)

    ([(segment_start_datetime[i], segment_end_datetime[i])  for i in range(len(segment_start_datetime))])


    for isegment in range(len(segment_start_datetime)):
        start_time, end_time = segment_start_datetime[isegment], segment_end_datetime[isegment]
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        segment_size = end_time - start_time
        
        tmp = segment_size.astype('timedelta64[h]')
        print(f'Segment {isegment} : {tmp}')
        print(start_time)
        print(end_time)
        print(' ' )





def das2sac():
    sacpath = 'L:\\anyuan_mine_2_sac\\'
    if not os.path.exists(sacpath):  os.makedirs(sacpath)





def segment(segment_start_datetime , segment_end_datetime ,  save_path ,fname_npdatetime ,datafile):
    ch1 = 90
    ch2 = 1442 
    das_ch_id = np.arange(ch1, ch2)
    nw = 'AY'
    sta = 'DAS'
    mlist = np.array([2,5])
    nprocs = 20

    chunk_size = np.timedelta64(7200, 's') #increment in hours
    increment = np.timedelta64(120, 's') # one-time increment of data one thread holds


    interval = increment * nprocs # total increment of data all threads hold
    for isegment in range(len(segment_start_datetime)):
        
        start_time, end_time = segment_start_datetime[isegment], segment_end_datetime[isegment]
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        segment_size = end_time - start_time
        chunk_num = segment_size // chunk_size + 1
        
        print('Segment id: %d'%(isegment))
        print('Segment size: %s'%(segment_size.astype('timedelta64[s]'))) 
        print('Regular chunk size: %s'%(chunk_size))
        print('Regular chunk number: %s'%(chunk_num - 1)) 
        print('Remainder chunk size: %s'%((segment_size % chunk_size).astype('timedelta64[s]')))
        
        print('nCPU number: %s'%(nprocs)) 
        print('Increment by each worker: %s'%(increment)) 
        print('Interval size: %s'%(interval)) 
        print('Interval number of regular chunks (if any): %s'%(chunk_size // interval))
        print('Interval number of the remainder chunk: %s'%(segment_size % chunk_size // interval))
        print(' ')
        
        segment_folder_path = sacpath + 'SAC-segment-' + ''.join(str(start_time.astype('datetime64[s]')).split(':')) 
        if not os.path.exists(segment_folder_path):
            os.makedirs(segment_folder_path)

        for ichunk in range(chunk_num):

            chunk_start_time = start_time + chunk_size * ichunk

            chunk_folder_path = os.path.join(segment_folder_path, 
                                            'SAC-chunk-' + ''.join(str(chunk_start_time.astype('datetime64[s]')).split(':')))

            if os.path.exists(chunk_folder_path):
                print('Chunk %d of Segment %d folder already exists: %s'%(ichunk, isegment, chunk_folder_path))
                print(' ')
                continue
                
            print('Chunk %d in Segment %d: %s'%(ichunk, isegment, chunk_folder_path))
            print('Start time of this chunk: %s'%(chunk_start_time.astype('datetime64[s]')))

            data = []
            interval_num = chunk_size // interval if ichunk < chunk_num-1 else segment_size % chunk_size // interval
            print('Total intervals in this Chunk: %d'%(interval_num))
            with Pool(processes=nprocs) as pool:
                for j in range(int(interval_num)):
                    print('Working on Interval %d in Chunk %d in Segment %d'%(j, ichunk, isegment))
                    increment_start_times = [ chunk_start_time + (interval * j) + (increment * i) for i in range(nprocs) ]
                    res = pool.map(partial(das_processing, 
                                        interval=increment, 
                                        datafile=datafile, 
                                        datafile_time=fname_npdatetime, 
                                        ch1=ch1, ch2=ch2, mlist=mlist), 
                                increment_start_times)

                    data.append(np.concatenate([res[i][0] for i in range(len(res))], axis=1))
                    dt = res[0][1]
                    del res

            if len(data) > 0:
                data = np.concatenate(data, axis=1)

                chunk_start_time_1 = chunk_start_time + increment
                datafile_arg_choose = np.where((fname_npdatetime>=chunk_start_time)&(fname_npdatetime<chunk_start_time_1))[0]
                chunk_start_time_from_file = fname_npdatetime[datafile_arg_choose][0]

                das_st = Stream()
                for ich in das_ch_id:

                    data_ich = np.where(das_ch_id==ich)[0][0]

                    tr = Trace(data=data[data_ich,:], header={'network':nw, 
                                                            'station': sta, 
                                                            'location':str(data_ich), 
                                                            'channel': str(ich),
                                                            'starttime':str(chunk_start_time_from_file), 
                                                            'delta':dt})

                    das_st.append(tr)

                print('Writing to sac...')
                
                if not os.path.exists(chunk_folder_path):
                    os.makedirs(chunk_folder_path)
                
                with Pool(processes=nprocs) as pool:
                    pool.map(partial(das_st_write_sac, date_folder_path=chunk_folder_path, write_coordinates=False), das_st)

                del tr
                del das_st
                del data
            gc.collect()
            print(' ')

#############################################################################################

def show_dir_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # THE LAST ITEM
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))
                

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # THE LAST ITEM
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))

###############################################################################################               
def concat(flist , start = 0 , span = 20 ):
    datalist = []
    for fname in flist[ start : min( start + span , len(flist)) ]:
        data = dp.read_das(fname)
        datalist.append(data)

    data = np.concatenate(datalist, axis = 1)
    utc_datetime = datetime.datetime.strptime(flist[start][-22 : -7],'%Y%m%d_%H%M%S') + datetime.timedelta(hours=+8)
    utc_day = utc_datetime.strftime("%m-%d %H:%M")
    return data , utc_datetime, utc_day 

def cal_and_save_psd_of_chunk(h5file , startn , endn , flist ,  ch_list , nfft , fs ):
    with h5py.File(h5file, 'a') as f:
    # 创建一个dataset
        span_each_pic = 10 #12h -> 50min
        concat_time = 0
        for n in range(startn, endn):
            data , utc_datetime, utc_day = concat(flist , start= span_each_pic * n ,span= span_each_pic)
            start = time.time()
            for ch in ch_list:
                if not f.__contains__(str(ch)):   #判断这个组存不存在
                    g = f.create_group(str(ch))
                else:
                    g = f[ch]
                [Pxx1,f1] = signal.welch(data[ch],         # 随机信号
                                nfft=nfft,               # 每个窗的长度
                                fs=fs,                   # 采样频率   
                                # detrend='mean',          # 去掉均值
                                window=np.hanning(nfft), # 加汉尼窗
                                noverlap=int(nfft*3/4),  # 每个窗重叠75%的数据
                                                      )        # 求单边谱
                g.create_dataset(str(utc_day),  data=Pxx1)
            concat_time +=  time.time() -start
            print(n , "拼接运行时间:%.2f min %d s"%( int(concat_time / 60) , concat_time%60) , datetime.datetime.now() , utc_day, "check in ")
        f.close()


##show the temporal variation of ch in chlist in fq domain  
def show_psd_Hz_time(psd_flist ,ch_list , frequencise,  ):
    for ch in ch_list:
        psd = []
        all_time = []
        for fname in psd_flist:
            with h5py.File(fname, 'r') as f:
                if len(f[str(ch)].keys()) == 1 :
                    dset = f[str(ch)]['01']
                else:
                    dset = f[str(ch)]
                time = [key for key in dset.keys()]

                psd.append(  [dset[key][:] for key in time][:])
                all_time.append(time[:])
                f.close()
        print("-----------read",ch,"------------")
        psd = np.log(np.concatenate(psd))
        time = np.concatenate(all_time)
        plt.figure(figsize=(30,10))
        plt.imshow(psd.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
        plt.grid(alpha = 1)
        xstick = range(0, len(time) , 20)
        ystick = [frequencise[-1]*30*i for i in range(10)]
        plt.xticks(xstick , ["1."+time[i][-7]+" "+time[i][-5:]  for i in xstick],rotation = 0)
        plt.yticks(ystick , ['%d'% (i/30.)  for i in ystick])
        plt.ylabel("Frequency(Hz)")
        plt.xlabel("Time")
        plt.title("log(PSD) Date=1.3-1.5 Channel="+str(ch)+" nfft=30,000")

        plt.colorbar()
        plt.savefig('../output/DAS/psd_5_9/ch_'+str(ch)+'_psd_.png') #10s per pic
        plt.close()

def show_psd_Hz_km(flist  , ch_list, frequencise ,  dis_spacing ,  title ,  save_path , log = True , start_time = None , end_time = None ):
    all_psd = []
    for ch in ch_list:
        psd = []
        for fi in flist:
            with h5py.File(fi, 'r') as f:
                if len(f[str(ch)].keys()) == 1 :
                    dset = f[str(ch)]['01']
                else:
                    dset = f[str(ch)]
                time = dset.keys()
                if start_time is not None & end_time is not None : 
                    time = time[time.index(start_time):time.index(end_time)+1]
                
                psd.append(  [dset[key][:] for key in time])
                f.close()
        
        if len(flist) > 1 : 
            psd = np.concatenate(psd)
        all_psd.append(np.log(np.mean(psd,axis=1))[0])
    all_psd= np.array(all_psd)
    plt.figure(figsize=(30,8))
    plt.imshow(all_psd.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
    xstick = np.array(range(0, len(ch_list) * dis_spacing , 125 ) ) /2
    ystick = [frequencise[-1]*30*i for i in range(10)]
    plt.xticks(xstick ,[ch_list[i]  for i in xstick] ,rotation = 0)
    plt.yticks(ystick , ['%d'% (i/30.)  for i in ystick])
    plt.ylabel("Frequency")
    plt.xlabel("Channel")
    plt.grid()

    plt.title("log(mean(PSD))"+title+ start_time +"-"+ end_time)
    plt.colorbar()
    plt.savefig(save_path)
    plt.show()
    plt.close()


#check
def channel_chunk_show_psd_Hz_time(psd_path ,ch_list, frequencise, title ,save_path , timeshift = None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ch in ch_list:
        psd = []
        all_time = []
        with h5py.File(psd_path+str(ch)+'.hdf5', 'r') as f:
            
            time = [key for key in f.keys()]
            #日期的排序和字符排序问题
            if timeshift is not None:
                start = time.index(timeshift)
                time = time[start:]+time[:start]
            psd.append([f[key][:] for key in time])
            all_time.append(time[:])
            f.close()
        print("-----------read",ch,"------------")
        
        psd = np.concatenate(psd)
        time = np.concatenate(all_time)
        psd = np.log(psd)
        plt.figure(figsize=(30,10))
        plt.imshow(psd.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
        plt.grid(alpha = 1)

        time_show = time[::40]
        plt.xticks(np.linspace(0 , len(psd)  , len(time_show),endpoint= False) , [t[-8:] for t in time_show],rotation = 0)

        plt.yticks(np.linspace(0 , len(psd[0])  , 11 ), np.linspace(0,fs/2 , 11 ,dtype=np.int16))
        plt.ylabel("Frequency(Hz)")
        plt.xlabel("Time")
        plt.title(title+" Channel="+str(ch))

        plt.colorbar()
        plt.show()
        plt.savefig(save_path+'ch_'+str(ch)+'_psd_.png') #10s per pic
        plt.close()

#check
def channel_chunk_show_psd_Hz_time_split(psd_path ,ch_list, frequencise, title ,save_path , timeshift = None , split = None , time_ticks = 15):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ch in ch_list:
        psd = []
        all_time = []
        with h5py.File(psd_path+str(ch)+'.hdf5', 'r') as f:
            
            time = [key for key in f.keys()]
            #日期的排序和字符排序问题
            if timeshift is not None:
                start = time.index(timeshift)
                time = time[start:]+time[:start]
            psd.append([f[key][:] for key in time])
            all_time.append(time[:])
            f.close()
        print("-----------read",ch,"------------")
        psd = np.concatenate(psd)
        time = np.concatenate(all_time)
        print(psd.shape)
        flim= len(psd[0])//2+2
        psd = psd[: , :flim]
        
        psd = np.log(psd)
        print(psd.shape)

        psdlist = np.array_split(psd , split ,axis=0)
        timelist = np.array_split(time ,split)

        for i in range(split):
            psdi = psdlist[i]
            timei = timelist[i]
            
            plt.figure(figsize=(30,10))
            plt.imshow(psdi.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
            plt.grid(alpha = 1)

            timei_show = range(0 , len(timei) , time_ticks )
            
            plt.xticks(timei_show , [timei[t][-8:] for t in timei_show],rotation = 0)

            plt.yticks(np.linspace(0 , len(psdi[0])  , 11 ), np.linspace(0,fs/2 , 11 ,dtype=np.int16))
            plt.ylabel("Frequency(Hz)")
            plt.xlabel("Time")
            plt.title(title+" Channel="+str(ch))

            plt.colorbar()
            plt.savefig(save_path+'ch_'+str(ch)+" " +str(i)+'_psd_.png') #10s per pic
            plt.close()

def channel_chunk_show_psd_Hz_time_split_(psd_path ,ch_list, frequencise, title ,save_path , timeshift = None , split = None , time_ticks = 15):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ch in ch_list:
        psd = []
        all_time = []
        with h5py.File(psd_path+str(ch)+'.hdf5', 'r') as f:
            
            time = [key for key in f.keys()]
            #日期的排序和字符排序问题
            if timeshift is not None:
                start = time.index(timeshift)
                time = time[start:]+time[:start]
            psd.append([f[key][:] for key in time])
            all_time.append(time[:])
            f.close()
        print("-----------read",ch,"------------")
        psd = np.concatenate(psd)
        time = np.concatenate(all_time)
        print(psd.shape)
        flim= len(psd[0])//2+2
        psd = psd[: , :flim]
        
        psd = np.log(psd)
        print(psd.shape)

        psdlist = np.array_split(psd , split ,axis=0)
        timelist = np.array_split(time ,split)

        for i in range(split):
            psdi = psdlist[i]
            timei = timelist[i]
            
            plt.figure(figsize=(30,10))
            plt.imshow(psdi.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
            plt.grid(alpha = 1)

            timei_show = range(0 , len(timei) , time_ticks )
            
            plt.xticks(timei_show , [timei[t][-8:] for t in timei_show],rotation = 0)

            plt.yticks(np.linspace(0 , len(psdi[0])  , 11 ), np.linspace(0,fs/2 , 11 ,dtype=np.int16))
            plt.ylabel("Frequency(Hz)")
            plt.xlabel("Time")
            plt.title(title+" Channel="+str(ch))

            plt.colorbar()
            plt.savefig(save_path+'ch_'+str(ch)+" " +str(i)+'_psd_.png') #10s per pic
            plt.close()
#check
def cal_and_show_stft(flist ,title, nperseg , fs , overlap , save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_start = time.time()
    for fname in flist:

        with h5py.File(fname, 'r') as f:
            time_list = [key for key in f.keys()]

            data= [f[key][:] for key in time_list][:]
            data = np.concatenate(data)

            f1, t1, Spg = signal.spectrogram(data, fs, nperseg = nperseg, noverlap=overlap )
            # f.create_dataset(str(utc_day),  data=Spg ) #save
            # f.close()
            # with h5py.File('../output/DAS/data_5_12/test3_ft.hdf5', 'w') as f2:
            #     f2.create_dataset("f",  data=f1 )
            #     f2.create_dataset("t",  data=t1 )

            Spg = Spg[:53]

            # for spgi in Spg[::]
            ch = fname.split('/' )[-1][:-5]
            print(ch)
            Spg = np.log10(Spg) #plot
            plt.figure(figsize=[18,3])
            vmin = np.min(Spg)+0.618*(np.max(Spg)-np.min(Spg))
            vmax = np.max(Spg)-(np.max(Spg)-np.min(Spg))/5
            plt.imshow(Spg[:,:], aspect='auto', cmap='jet', vmin=0, vmax=vmax+3, origin='lower', extent=[t1[0], t1[-1], f1[0], f1[53]])
            xticks = [i for i in  range(0 , len(t1) , (len(t1) // (len(time_list )+1) *12))]
            plt.xticks([t1[i] for i in xticks][:18], [t[-8:-3] for t in time_list[::12]])
            plt.ylabel("Frequency(Hz)")
            plt.xlabel("Time")
            plt.title(ch + title + time_list[0][:-8])
            plt.colorbar()
            plt.grid()
            plt.savefig(save_path+ch)
            plt.show()
            plt.close()

        all_time = time.time() -all_start
        print("运行时间 all :%.2f min %d s"%( int(all_time / 60) , all_time%60) )

#check
def cal_and_show_stft_split(flist ,title, nperseg , fs , overlap , split , save_path, channel_rete = 5 ,figsize0 = [15,6]  , lp = 0 , hp = 1):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_start = time.time()
    for fname in flist:

        with h5py.File(fname, 'r') as f:
            all_time = np.array( [key for key in f.keys()])

            data= [f[key][:] for key in all_time][:]
            data = np.concatenate(data)

            f1, t1, Spg = signal.spectrogram(data, fs, nperseg = nperseg, noverlap=overlap )
            # f.create_dataset(str(utc_day),  data=Spg ) #save
            # f.close()
            # with h5py.File('../output/DAS/data_5_12/test3_ft.hdf5', 'w') as f2:
            #     f2.create_dataset("f",  data=f1 )
            #     f2.create_dataset("t",  data=t1 )

            llim = int(len(f1)*lp)
            hlim= int(len(f1)*hp)+2
            Spg = Spg[llim:hlim] #only 0-100Hz
            
            # for spgi in Spg[::]
            ch = fname.split('/' )[-1][:-5]+"  "
            print(ch , Spg.shape , all_time.shape)
            Spg = np.log10(Spg) #plot

            Spglist = np.array_split(Spg , split ,axis=1)
            t1list = np.array_split(t1 , split)
            timelist = np.array_split(all_time ,split)
            for s in range(0, split):
                Spgi , t1i , timei = Spglist[s] , t1list[s] , timelist[s]
                plt.figure(figsize=figsize0)
                vmin = np.min(Spgi)+0.618*(np.max(Spgi)-np.min(Spgi))
                vmax = np.max(Spgi)-(np.max(Spgi)-np.min(Spgi))/5
                plt.imshow(Spgi[:,:], aspect='auto', cmap='jet', vmin=0, vmax=vmax+3, origin='lower', extent=[t1i[0], t1i[-1], f1[0], f1[hlim]])
                timei_show = timei[::6] 
                plt.xticks(np.linspace(t1i[0], t1i[-1] , len(timei_show) , endpoint = False ), [t[-8:-3] for t in timei_show])
                plt.ylabel("Frequency(Hz)")
                plt.xlabel("Time")
                plt.title(ch + title +"  "+ timei[0][:-8])
                plt.colorbar()
                plt.grid()
                plt.savefig(save_path+ch+str(s)+" "+str(nperseg) +" "+str(overlap))
                # plt.show()
                plt.close()

        all_time = time.time() -all_start
        print("运行时间 all :%.2f min %d s"%( int(all_time / 60) , all_time%60) )

def show_stft_time(flist ,ft, ch_list,  start_time , end_time, title, save_path):
    
    index = 0 
    frequencies = h5py.File(ft , 'r')['f']
    N_window = h5py.File(ft , 'r')['t']

    for fnmae in flist:
        stft = []
        all_time = []
        with h5py.File(fnmae, 'r') as f:
            
            time = [key for key in f.keys()][:4]
            #日期的排序和字符排序问题
            
            # start = time.index('12-30 21:55')
            # time = time[start:]+time[:start]
            # print(time)
            # start = time.index('20:00:12')
            # time = time[start:]+time[:start]
            # print(time)
            stft.append(  [f[key][:].T for key in time])
            all_time.append(time[:])
            f.close()

        ch = ch_list[index]
        print("-----------read",ch,"------------")
        stft = np.log(np.array(stft).reshape(-1 , len(frequencies)))
        print(len(all_time))    
        # stft = np.vstack(stft )

        time = np.concatenate(all_time)
        print(stft.shape)
        print(len(time))
        stft = np.log(stft)
        plt.figure(figsize=(60,10))
        plt.imshow(stft.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
        # plt.imshow(stft.T, aspect='auto', cmap='viridis')
        plt.grid(alpha = 1)

        

        xstick = range(0, len(time) *len(N_window), len(N_window))
        print(xstick)
        ystick = range(0, len(frequencies) , 10)
        print(ystick)
        # print(f['frequency'].shape)

        plt.xticks(xstick ,time,rotation = 30)
        # plt.xticks(xstick , [time[i][:2]+" "+time[i][3:]  for i in xstick],rotation = 0)

        plt.yticks(ystick , ['%d'% (frequencies[i])  for i in ystick])
        plt.ylabel("Frequency")
        plt.xlabel("Time")
        plt.title("log(stft)"+title+" "+ start_time+"-"+end_time)

        plt.colorbar()
        plt.savefig(save_path+'ch_'+str(ch)+'_stft_.png') #10s per pic
        plt.close()
        index += 1

def show_stft_Hz_time():
    stft = []
    all_time = []
    for fnmae in flist:
        with h5py.File(fnmae, 'r') as f:
            
            time = [key for key in f.keys()]
            #日期的排序和字符排序问题
            
            # start = time.index('12-30 21:55')
            # time = time[start:]+time[:start]
            # print(time)
            # start = time.index('20:00:12')
            # time = time[start:]+time[:start]
            # print(time)

            stft.append(  [f[key][:] for key in time])
            all_time.append(time[:])
            f.close()

        ch = fnmae.strip 
        print("-----------read",ch,"------------")
    
    print(len(stft))
    print(len(all_time))    
    stft = np.concatenate(stft)

    time = np.concatenate(all_time)
    print(stft.shape)
    print(len(time))
    stft = np.log(stft)
    plt.figure(figsize=(60,10))
    plt.imshow(stft.T, aspect='auto', cmap='jet',vmin=0, vmax=12)
    # plt.imshow(stft.T, aspect='auto', cmap='viridis')
    plt.grid(alpha = 1)

    xstick = [i for i in range(1, len(time) , 12)]
    print(xstick)
    ystick = [1500*i for i in range(10)]
    # print(f['frequency'].shape)

    plt.xticks(xstick , [time[i][-11:-3]+'h'  for i in xstick],rotation = 30)
    # plt.xticks(xstick , [time[i][:2]+" "+time[i][3:]  for i in xstick],rotation = 0)

    plt.yticks(ystick , ['%d'% (i/15.)  for i in ystick])
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.title("log(stft)"+title+" "+ start_time+"-"+end_time)

    plt.colorbar()
    plt.savefig(path+'ch_'+str(ch)+'_stft_.png') #10s per pic
    plt.close()

def find_mutations(data, threshold):
    mutations = []
    for i in range(2, len(data)):
        slope_diff = abs((data[i] / data[i-2]) / (i - (i-1)))
        if slope_diff > threshold:
            mutations.append(i)
    for i in range(0, len(data)-2):

        slope_diff = abs((data[i] / data[i+2]) / (i - (i+1)))
        if slope_diff > threshold:
            mutations.append(i)
    return mutations

def show_sum_and_psd(h5file , ch_list , dis_spacing , title , save_path , darkline = None , mutation_th = None , mutations = None):
    #tested

    with h5py.File(h5file, 'r') as f:
        print(f.keys())
        all_psd = f['all_psd']
        sum_psd = np.sum(all_psd , axis = 1)

        fig = plt.figure(figsize=(60,12))
        plt.subplot(211)
        log_sum_psd = np.log(sum_psd)

        


        plt.plot(log_sum_psd)
        xstick = np.array(range(0, len(ch_list) * dis_spacing , 125 ) ) /2
        plt.xticks(xstick ,[ i*8/1000  for i in xstick] ,rotation = 0)
        plt.xlim(0,len(log_sum_psd))
        plt.ylabel("Amplitude")
        plt.xlabel("Distance (km)")
        plt.grid()
        if darkline is not None:
            darkline = np.array(darkline) *125
            for i in darkline:
                plt.vlines(i, 0, 20, linestyles='dashed', colors='black')
        if mutation_th is not None:
            mutations = find_mutations(sum_psd , mutation_th)
            print(mutations)
        if mutations is not None:
            for i in mutations:
                plt.vlines(i, 0, 20, linestyles="dotted", colors='red')
        plt.title("log(sum(PSD))  " + title)

        

        plt.subplot(212)
        log_psd  = np.log(all_psd)
        im = plt.imshow(log_psd.T, aspect='auto', cmap='jet',vmin=-10, vmax=20)
        # plt.imshow(psd.T, aspect='auto', cmap='viridis')

        xstick = np.array(range(0, len(ch_list) * dis_spacing , 125 ) ) /2
        ystick = [150*i for i in range(11)]

        xstick_2 = np.append(xstick , mutations)
        xstick_3 = np.append(xstick_2 , darkline)

        plt.xticks(xstick_3 ,[ "%d"%(i*dis_spacing) for i in xstick_3] ,rotation = 90)
        # for i in mutations:
        #     plt.get_xticklabels()[i].set_color("red")


        # plt.xticks(xstick ,[ i  for i in xstick] ,rotation = 0)

        plt.yticks(ystick , ['%d'% (i/6.)  for i in ystick])
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Distance (m)")
        plt.grid(visible=1)
        if darkline is not None:
            for i in darkline:
                plt.vlines(i, 0, 1500, linestyles='dashed', colors='black')
        if mutations is not None:
            for k in mutations:
                plt.vlines(k, 0, 1500, linestyles="dotted", colors='red')
        plt.title("log(mean(PSD))  "+title)
        position=fig.add_axes([0.92, 0.3, 0.01, 0.4])#位置[左往右移动,下往上移动,宽度压缩,高度压缩]
        cb=plt.colorbar(im,cax=position)#方向

        # plt.colorbar()
        plt.savefig(save_path) #10s per pic
        plt.show()
        plt.close()


#check
def show_concat(flist , start, stop , step ,decimate , clim_rate ,title , save_path ,N):
    #tested
    for n in range(start, stop , step ):
        index = str(n//stop)
        datalist , timelist = [] , [] 

        for fname in flist[ n : min( n + step , len(flist)) ]:
            data = dp.read_das(fname)
            datalist.append(data)
            utc_datetime = datetime.datetime.strptime(fname[-22 : -7],'%Y%m%d_%H%M%S') + datetime.timedelta(hours=+8)
            timelist.append(utc_datetime)

        data = np.concatenate(datalist, axis = 1)
        if decimate > 1 :
            data = signal.decimate(data, decimate, axis = 1)
        utc_day = utc_datetime.strftime("%m-%d")

        plt.figure(figsize=[16,10])
        plt.title(title)

        clim= data.std() 
        d_clim = data.std() / clim_rate
        # norm = matplotlib.colors.Normalize(vmin=-clim, vmax=+clim)
        xticks = range(0,len(timelist) * N , N)
        print(clim , len(timelist) , data.shape)
        plt.grid()
        plt.xticks(xticks ,timelist,rotation = 20, ha = 'right')
        # plt.yticks(range(0,80,10),range(820,900,10))
        plt.imshow(data, aspect='auto', cmap='RdBu', vmin=-d_clim, vmax=d_clim )
        # plt.imshow(data, aspect='auto', cmap='RdBu', norm=norm )
        plt.xlabel("Time")
        plt.ylabel("Channel")
        plt.colorbar()
        plt.savefig(save_path +"concat_"+index)
        plt.close()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 

def normalization2(data):
    _range = np.max(data) - np.min(data)
    return (data) / _range *10
 
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

#check
def show_concat_channel_balenced(flist ,psdh5file, start, stop , step ,decimate , clim_rate ,title , save_path ,N ):

    sum_psd = h5py.File(psdh5file, 'r')['sum_psd'][:].reshape(-1,1)
    norm_sum_psd = np.divide(sum_psd , np.max(sum_psd)/147 ) #平衡后的数据 整体分布区间要和原来一致
    #老师，das直接除以sum psd值会太小，把sum psd做归一化会有零值，为了使得平衡后的整体值和之前相近，我先将das data 乘了一个系数，再除以sum data

    for n in range(start, stop , step ):
        index = str(n//step)
        datalist , timelist = [] , [] 

        for fname in flist[ n : min( n + step , len(flist)) ]:
            data = dp.read_das(fname)
            datalist.append(data)
            utc_datetime = datetime.datetime.strptime(fname[-22 : -7],'%Y%m%d_%H%M%S') + datetime.timedelta(hours=+8)
            timelist.append(utc_datetime.strftime("%H:%M %Ss"))

        data = np.concatenate(datalist, axis = 1)
        balance_data = np.divide(data , norm_sum_psd )
        if decimate > 1 :
            balance_data = signal.decimate(data, decimate, axis = 1)
        utc_day = utc_datetime.strftime("%m-%d")

        plt.figure(figsize=[32,15])
        plt.rcParams.update({"font.size":20})
        plt.subplot(121)
        plt.title(title+"before "+utc_day)

        clim= data.std() 
        d_clim = data.std() / clim_rate
        xticks = range(0,len(timelist) * N , N)
        print(clim , len(timelist) , data.shape)
        plt.grid()


        plt.xticks(xticks ,timelist,rotation = 0, ha = 'left')
        plt.imshow(data, aspect='auto', cmap='RdBu', vmin=-d_clim, vmax=d_clim )
        plt.xlabel("Time")
        plt.ylabel("Channel")
        plt.colorbar()


        plt.subplot(122)
        plt.title(title+"after "+utc_day)
        clim= balance_data.std() 
        d_clim = balance_data.std() / clim_rate
        xticks = range(0,len(timelist) * N , N)
        print(clim , len(timelist) , balance_data.shape)
        plt.grid()

        plt.xticks(xticks ,timelist,rotation = 0, ha = 'left' ,fontsize=20)
        plt.imshow(balance_data, aspect='auto', cmap='RdBu', vmin=-d_clim, vmax=d_clim )
        plt.xlabel("Time")
        plt.ylabel("Channel")
        plt.colorbar()
        plt.show()
        plt.savefig(save_path+"channel_balanced_" +index)
        plt.close()


#check
def channel_2_wav(flist , save_path , fs ,  start_time = None  ,end_time =None ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for fname in flist:

        with h5py.File(fname, 'r') as f:
            time_list = [key for key in f.keys()]
            print(time_list[:8])
            if start_time is not None and end_time is not None : 
                    time_list = time_list[time_list.index(start_time):time_list.index(end_time)+1]
            data= [f[key][:] for key in time_list][:]
            data = np.concatenate(data)
            ch = fname.split('/' )[-1][:-5]
            scipy.io.wavfile.write(save_path+ch+".wav" , fs ,data)

#check
def das_time2ch(flist , ch_list ,  start,stop , step ,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for n in range(start, stop , step):
        data , utc_datetime, utc_day = concat(flist , start= n ,span= step)
        start = time.time()
        print(n, data.shape)
        for ch in ch_list:
            with h5py.File(save_path+str(ch)+'.hdf5', 'a') as f:
                f.create_dataset(str(utc_datetime),  data= data[ch] )



def show_ifft_time_point_amtitude_channel(flist  , ch_list, frequencise ,  dis_spacing ,  title ,  save_path  , timeindex = [0] , start_time = None , end_time = None ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ti in timeindex:
        all_AC = []
        for ch in ch_list:
            for fi in flist:
                with h5py.File(fi, 'r') as f:
                    if len(f[str(ch)].keys()) == 1 :
                        dset = f[str(ch)]['01']
                    else:
                        dset = f[str(ch)]
                    time = [i for i in dset.keys()]
                    
                    if start_time is not None and end_time is not None : 
                        time = time[time.index(start_time):time.index(end_time)+1]
                    timei = time[ti]
                    psd = np.array(dset[timei])
                    print(psd.shape)
                    AC = np.fft.ifft( np.abs(psd))
                    f.close()
                    # AC = np.fft.fftshift(AC)
                    all_AC.append(abs(AC))
                    print(len(all_AC))

        all_AC= np.array(all_AC)
        plt.figure(figsize=(30,8))
        plt.imshow(np.log(all_AC).T, aspect='auto', cmap='jet', vmin=0, vmax=np.max(np.log(all_AC)))
        print(np.max(np.log(all_AC)))
        # xstick = np.array(range(0, len(ch_list) * dis_spacing , 125 ) ) //2
        ystick = [frequencise[-1]*3*i for i in range(10)]
        # plt.xticks(xstick , [ch_list[i]  for i in xstick] ,rotation = 0)
        plt.yticks(ystick , ['%d'% (i/30.)  for i in ystick])
        plt.ylabel("Frequency")
        plt.xlabel("Channel")
        plt.grid()
        plt.title("log(ifft(PSD))"+title + timei)
        plt.colorbar()
        plt.savefig(save_path+timei)
        plt.show()
        plt.close()


######
#   

def show_ifft_time_range_amtitude_channel(flist  , ch_list, frequencise ,  dis_spacing ,  title ,  save_path  , t_range = [0,-1] , start_time = None , end_time = None  ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_AC = []
    all_psd = []
    for ch in ch_list:
        for fi in flist:
            with h5py.File(fi, 'r') as f:
                if len(f[str(ch)].keys()) == 1 :
                    dset = f[str(ch)]['01']
                else:
                    dset = f[str(ch)]
                time = [i for i in dset.keys()][t_range[0] : t_range[-1]]
                
                if start_time is not None and end_time is not None : 
                    time = time[time.index(start_time):time.index(end_time)+1]
                psd = np.average([dset[ti] for ti in time] ,axis=0)
                all_psd.append(psd)
                # print(psd.shape)
                # plt.figure()
                # plt.plot(np.append(psd[::-1][:-2] , psd))
                # print(np.append(psd[::-1][:-2] , psd)[15000-3:15000+3])
                # plt.show()
                AC = np.fft.ifft( np.abs(np.append(psd[::-1][:-2] , psd)))
                # AC2 = np.fft.ifft( np.abs( psd))
                # # avg_AC = np.average([np.fft.ifft( np.abs(np.append(np.array(dset[ti])[::-1][:-2] , dset[ti]))) for ti in time] ,axis=0)
                # plt.figure()
                # plt.plot(AC)
                # plt.plot(avg_AC)
                # plt.plot(AC2)
                # plt.show()
                # print(AC.shape)
                f.close()
                # AC = np.fft.fftshift(AC)
                all_AC.append(np.abs(AC[: len(AC)//2]))

    if False:
        with h5py.File(save_path+'acf' +'.hdf5', 'w') as f:
            f.create_dataset(str(time[0] + " - "+time[-1]),  data=all_AC )
            f.close()
    # all_psd = all_psd[80:]
    # all_AC = all_AC[80:]
 
    all_psd= np.array(all_psd)
    all_psd_s= np.log(all_psd)
    
    total_fig = 700

    plt.figure(figsize=(30, 30 ))
    plt.tight_layout()
    plt.subplot(total_fig+11)
    clim = all_psd_s.std() /30
    plt.imshow(all_psd_s.T, aspect='auto', cmap='jet', vmin=0, vmax=12)
    ystick = [frequencise[-1]*3*i for i in range(10)]
    plt.yticks(ystick , ['%d'% (i/30.)  for i in ystick])
    plt.ylabel("fq(Hz)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title("log(PSD)"+title + time[0] + " - "+time[-1])
    plt.colorbar()


    sum_psd = np.sum(all_psd,axis = 1)
    plt.subplot(total_fig+12)
    plt.plot(sum_psd)
    plt.ylabel("amplitude")
    plt.xlabel("Channel")
    plt.grid()
    plt.title("sum(PSD)"+title + time[0] + " - "+time[-1])

    balenced_psd =  np.divide(all_psd.T,sum_psd /70000)
    balenced_psd_s = np.log(balenced_psd)

    plt.subplot(total_fig+13)
    clim = balenced_psd.std() 
    print(clim ,all_psd.std()) 
    plt.imshow(balenced_psd_s, aspect='auto', cmap='jet', vmin=0, vmax=6)
    ystick = [frequencise[-1]*3*i for i in range(10)]
    plt.yticks(ystick , ['%d'% (i/30.)  for i in ystick])
    plt.ylabel("fq(Hz)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title("balence (PSD)"+title + time[0] + " - "+time[-1])
    plt.colorbar()

    all_AC= np.array(all_AC)
    all_AC = np.divide(all_AC.T , sum_psd /70000 ).T
    all_AC_s= all_AC /np.max(all_AC)
    plt.subplot(total_fig+14)
    clim = all_AC_s.std() /2
    plt.imshow(all_AC_s.T, aspect='auto', cmap='jet', vmin=0, vmax=clim)
    ystick = [5000*i for i in range(4)]
    plt.yticks(ystick , ['%d'% (i/1000.)  for i in ystick])
    plt.ylabel("Time(s)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title("ifft(PSD)"+title + time[0] + " - "+time[-1])
    plt.colorbar()


    w = 7
    rolling_mean = np.array ([ np.convolve(ac, np.ones(w), "valid") / w for ac in all_AC] )
    rolling_mean_s = rolling_mean / np.max(rolling_mean)
    plt.subplot(total_fig+15)
    clim = rolling_mean_s.std() /2
    plt.imshow(rolling_mean_s.T, aspect='auto', cmap='jet', vmin=0, vmax=clim)
    ystick = [5000*i for i in range(4)]
    plt.yticks(ystick , ['%d'% (i/1000.)  for i in ystick])
    plt.ylabel("Time(s)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title(" moving window average (ifft(PSD) , w = 20 )"+title + time[0] + " - "+time[-1])
    plt.colorbar()


    deco = np.array([scipy.signal.deconvolve(all_AC[i] , rolling_mean[i])[1] for i in range(len(all_AC))])
    print(deco.shape)
    deco_s = deco/np.max(deco)
    plt.subplot(total_fig+16)
    clim = deco_s.std() /3
    plt.imshow(deco_s.T, aspect='auto', cmap='gray', vmin=0, vmax=clim)
    ystick = [5000*i for i in range(4)]
    plt.yticks(ystick , ['%d'% (i/1000.)  for i in ystick])
    plt.ylabel("Time(s)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title(" decon(rolliong(ifft(PSD))) "+title + time[0] + " - "+time[-1])
    plt.colorbar()

    filter_data = dp.bandpass(deco, 0.001, 1 , 30 )
    filter_data_s = filter_data/np.max(filter_data)
    plt.subplot(total_fig+17)
    clim = filter_data_s.std()/20
    plt.imshow(filter_data_s.T, aspect='auto', cmap='gray', vmin=0, vmax=clim)
    ystick = [5000*i for i in range(4)]
    plt.yticks(ystick , ['%d'% (i/1000.)  for i in ystick])
    plt.ylabel("Time(s)")
    plt.xlabel("Channel")
    plt.grid()
    plt.title("1-30 filter decon(rolliong(ifft(PSD))) "+title + time[0] + " - "+time[-1])
    plt.colorbar()
    plt.savefig(save_path+time[0]+"-"+time[-1]+" 1-30 filter,2  ")
    plt.show()
    plt.close()
