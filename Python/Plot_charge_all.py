import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = r"F:\data\20220314\5um_SiO2\1\discharge\5"
ts = 1.

fdrive = 35.  # 31.
make_plot = True

data_columns = [0, bu.xi]  # column to calculate the correlation against
drive_column = 3  # column containing drive signal


def getphase(fname):
    print
    "Getting phase from: ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]
    xdat = dat[:, data_columns[1]]
    xdat = xdat - np.mean(xdat)

    xdat = np.append(xdat, np.zeros(int(fsamp / fdrive)))
    corr2 = np.correlate(xdat, dat[:, drive_column] - np.mean(dat[:, drive_column]))
    maxv = np.argmax(corr2)

    cf.close()

    print ("maxv=",maxv)
    return maxv


def getdata(fname, maxv):
    print
    "Processing ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))

    if (len(attribs) > 0):
        fsamp = attribs["Fsamp"]

    xdat = dat[:, data_columns[1]]

    lentrace = len(xdat)
    ## zero pad one cycle
    MinusDC = dat[:, drive_column]
    print("AC=",1000*(np.max(MinusDC)-np.min(MinusDC))/2,"DC=",1000*(np.max(MinusDC)+np.min(MinusDC))/4)
    corr_full = bu.corr_func(MinusDC - np.median(MinusDC), xdat, fsamp, fdrive)

    # plt.figure()
    # plt.plot( xdat)
    # plt.plot(dat[:,drive_column])
    # plt.show()

    return corr_full[0], np.max(corr_full)


def get_most_recent_file(p):
    ## only consider single frequency files, not chirps
    filelist = glob.glob(os.path.join(p, "*.h5"))  ##os.listdir(p)
    filelist = sorted(filelist, key=os.path.getmtime)
    # filelist = [filelist[0]]
    mtime = 0
    mrf = ""
    for fin in filelist:
        if (fin[-3:] != ".h5"):
            continue
        f = os.path.join(path, fin)
        if os.path.getmtime(f) > mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fnum = re.findall('\d+.h5', mrf)[0][:-3]
    return mrf  # .replace(fnum, str(int(fnum)-1))


best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    # plt.hold(False)

last_file = ""
## get the most recent file in the directory and calculate the correlation

filelist = glob.glob(os.path.join(path, "*.h5"))
filelist = sorted(filelist, key=os.path.getmtime)
n=0

for cfile in filelist:

    if (not best_phase):
        best_phase = getphase(cfile)
    n=n+1
    print(n,'s')
    corr = getdata(cfile, best_phase)
    corr_data.append(corr)

np.savetxt(os.path.join(path, "current_corr.txt"), [corr, ])

if make_plot:
    plt.clf()
    plt.plot(np.array(corr_data))
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('Correlation value')
    plt.show()
