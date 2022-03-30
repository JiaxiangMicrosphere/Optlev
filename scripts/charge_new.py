
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
from scipy.fft import fft, ifft, rfft, fftfreq
import cPickle as pickle

path = r"F:\data\20220225\5um_SiO2\1\discharge\test"
ts = 1.

fdrive = 35. #31.
make_plot = True

data_columns = [0, bu.xi] # column to calculate the correlation against
drive_column = 3 # column containing drive signal

def getdata(fname):
    print ("Processing ", fname)
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))

    if( len(attribs) > 0 ):
        fsamp = attribs["Fsamp"]
    print ("Getting data from: ", fname) 
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]
    xdat = dat[:,data_columns[1]]# data of the displacement
    xdat = xdat - np.mean(xdat)
    fourier_x=rfft(xdat)#Do the Fourier transform of the x
    f = fftfreq( len(xdat),1/fsamp)
    freq_x = np.abs(f[0:int(len(xdat)/2)+1])
        
    data_drive=dat[:,drive_column] - np.mean(dat[:,drive_column]) # Electric field without offset
    fourier_drive=rfft(data_drive)  # Do the Fourier transform of the drive signal
    f = fftfreq( len(data_drive),1/fsamp)
    freq_drive = np.abs(f[0:int(len(data_drive)/2)+1])#frequency domain axis
    E_freq=freq_drive[np.where(fourier_drive==max(fourier_drive))[0]]

    print(E_freq)
                                                 
    n=np.where(np.abs(freq_x-E_freq)<0.5)
    maxv = max(np.abs(fourier_x[n]))
    maxf=np.where(np.abs(fourier_x)==maxv)[0]
    cf.close()
    return np.real(fourier_x[maxf][0]), np.real(maxv) 

def get_most_recent_file(p):

    ## only consider single frequency files, not chirps
    filelist = glob.glob(os.path.join(p,"*.h5"))  ##os.listdir(p)
    #filelist = [filelist[0]]
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fnum = re.findall('\d+.h5', mrf)[0][:-3]
    return mrf#.replace(fnum, str(int(fnum)-1))



if make_plot:
    fig0 = plt.figure()
    # plt.hold(False)

corr_data = []
last_file = ""
while( True ):
    ## get the most recent file in the directory and calculate the correlation

    cfile = get_most_recent_file( path )
    
    ## wait a sufficient amount of time to ensure the file is closed
    print (cfile)
    time.sleep(ts)

    if( cfile == last_file ): 
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep( 1 )


    corr = getdata( cfile )
    corr_data.append(corr )

    np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )

    if make_plot:
        plt.clf()
        plt.plot(np.array(corr_data))
        plt.grid()
        plt.draw()
        plt.pause(0.001)
