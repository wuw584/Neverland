import numpy as np
from scipy import signal
from nptdms import TdmsFile
import h5py
    

def read_tdms(filename):
    
    ### https://nptdms.readthedocs.io/en/stable/quickstart.html
    
    tdms_file = TdmsFile.read(filename)
    headers = tdms_file.properties

    nch = len(tdms_file['Measurement'])
    nt = len(tdms_file['Measurement']['0'])
    dt = 1./tdms_file.properties['SamplingFrequency[Hz]']
    dx = tdms_file.properties['SpatialResolution[m]']
    
    data = np.asarray([tdms_file['Measurement'][str(i)] for i in range(nch)])
    
    return data, headers, dt, dx


def read_h5(file):
    f = h5py.File(file,'r')
    data = f['Acquisition/Raw[0]/RawData/'][:].T
    dt = np.diff(f['Acquisition/Raw[0]/RawDataTime'][:]).mean()/1e6
    return data, dt


def das_preprocess(data_in):
    data_out = signal.detrend(data_in)
    data_out = data_out - np.median(data_out,axis=0)
    
    return data_out


def bandpass(data, dt, fl, fh):
    sos = signal.butter(6, [fl, fh], 'bp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


def highpass(data, dt, fl):
    sos = signal.butter(6, fl, 'hp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def lowpass(data, dt, fh):
    sos = signal.butter(6, fh, 'lp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


# def plot(data, fch, lch, 