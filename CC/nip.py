import numpy as np
from scipy import signal

from functools import partial
#定义了两个算互相关的函数，一个是算阵列ivs_arr和其他所有道的互相关，一个是算台站ivs和其他所有台站的互相关，
#算最近的两两台站的互相关应该就用或者改XCORR_vshot函数就行吧
def time_norm(data, dt, fl=5, fh=40):
    order = 6
    fNy = 0.5/dt
    sos = signal.butter(order, [fl/fNy, fh/fNy], analog=False, btype='band', output='sos')
    data_filt = signal.sosfilt(sos, data, axis=1)

    N = int(round(0.5/fl/dt))
    data_norm = np.zeros(data.shape)
    for i in range(data.shape[0]):
        weight = np.convolve(abs(data_filt[i,:]), np.ones((2*N+1,))/(2.*N+1.), mode='valid')
        weight = np.pad(weight, (N,N), 'edge')
        assert weight.shape[0]==data.shape[1]
        data_norm[i,:] = data[i,:]/(weight+1.e-20)

    return data_norm


def spec_norm(data):
    data_fft = np.fft.fft(data, axis=1)
    data_fft_norm = data_fft / (abs(data_fft) + 1e-20)
    data_norm = np.fft.ifft(data_fft_norm, axis=1)
    return data_norm.real


def XCORR_vshot(fname, dt, ivs, wlen):  
    data1 = np.load(fname, mmap_mode='r')
    
    nch, nt = data1.shape[0], data1.shape[1]
    nwin = nt // wlen
    
    XCF_out = np.zeros((nch, wlen*2-1))    
    for iwin in range(nwin):
        data1_vs = data1[ivs,(iwin*wlen):((iwin+1)*wlen)]
        
        XCF_out += np.asarray([signal.correlate(data1[ivr,(iwin*wlen):((iwin+1)*wlen)], data1_vs, mode='full', method='fft') 
                                  for ivr in range(nch)])
    return XCF_out
    
    
def XCORR_vshot_arr(fname, dt, ivs_arr, wlen):  
    data1 = np.load(fname, mmap_mode='r')
    
    nch, nt = data1.shape[0], data1.shape[1]
    nvs = len(ivs_arr)
    nwin = nt // wlen
    
    XCF_out = np.zeros((nvs, nch, wlen*2-1))
    for iwin in range(nwin):
        for ivs in range(nvs):
            data1_vs = data1[ivs_arr[ivs],(iwin*wlen):((iwin+1)*wlen)]
            XCF_out[ivs,:,:] += np.asarray([signal.correlate(data1[ivr,(iwin*wlen):((iwin+1)*wlen)], data1_vs, mode='full', method='fft') 
                                      for ivr in range(nch)])
    return XCF_out
    

ivs = 350
dt = 0.0025
wlen = int(5 / dt)
XCORR_partial = partial(XCORR_vshot, dt=dt, ivs=ivs, wlen=wlen)

# ivs_arr = np.arange(100,1601,40)
# ivs_arr = np.arange(360,1617,40)  # available channels range from 361 to 1617

ivs_arr = np.arange(370,1617,20)  # available channels range from 361 to 1617

XCORR_vshot_arr_partial = partial(XCORR_vshot_arr, dt=dt, ivs_arr=ivs_arr, wlen=wlen)

