from correlation import outputThetaPosition, getGainAndACamp, num_electrons_in_sphere
from VoltagevsAmplitude import conversion # gives N/V
import h5py, matplotlib, os, re, glob
import matplotlib.pyplot as plt
from bead_util import xi, drive
import numpy as np

# Inputs
NFFT = 2 ** 17
make_psd_plot = False
debugging = False

calib = "/data/20170622/bead4_15um_QWP/charge9"
path = "/data/20170622/bead4_15um_QWP/dipole27_Y"

if debugging:
    print "debugging on in plot_PSD_peaks.py: prepare for spam"
    print "num_electrons_in_sphere = ", num_electrons_in_sphere # 1E15
# in terminal, type 'python -m pdb plot_PSD_peaks.py'

def getdata(fname):
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    Fs = dset.attrs['Fsamp']
    dat = dat * 10. / (2 ** 15 - 1)
    x = dat[:, xi] - np.mean(dat[:, xi])
    xpsd, freqs = matplotlib.mlab.psd(x, Fs=Fs, NFFT=NFFT)
    drive_data = dat[:, drive] - np.mean(dat[:, drive])
    normalized_drive = drive_data / np.max(drive_data)
    drivepsd, freqs = matplotlib.mlab.psd(normalized_drive, Fs=Fs, NFFT=NFFT)
    return freqs, np.sqrt(xpsd), np.sqrt(drivepsd) # Hz, V/sqrtHz, 1/sqrtHz

def time_ordered_file_list(path):
    file_list = glob.glob(path + "/*.h5")
    file_list.sort(key=os.path.getmtime)
    return file_list

def get_positions(xpsd, dpsd):
    """ returns position of drive frequency and twice the drive frequency """
    tolerance = 3 # bins
    a = np.argmax(dpsd) # drive frequency bin
    b = 2*a
    if debugging:
        print ""
        print "DEBUGGING: get_positions"
        print "           len(xpsd) = ", len(xpsd)
        print "           a = ", a
        print "           b = ", b
    c = np.argmax(xpsd[b-tolerance:b+tolerance])
    if debugging:
        print "           c = ", c
    d = (b - tolerance) + c # twice drive frequency bin
    if debugging:
        print "           d = ", d
        print ""
    return a, d

def get_peak_amplitudes(xpsd, dpsd):
    """ This is Fernando's weird way of averaging the peak amplitudes """
    a, d = get_positions(xpsd, dpsd)
    peaksD = dpsd[a] # amplitude of drive
    peaks2F = xpsd[d] + xpsd[d - 1] + xpsd[d + 1]  # all 2F peak bins
    return peaks2F/peaksD, d

def plot_peaks2F(path, plot_peaks = True):
    file_list = time_ordered_file_list(path)
    amplitudes = []
    theta = []
    y_or_z = ""
    if make_psd_plot: plt.figure()
    for f in file_list:
        freqs, xpsd, dpsd = getdata(f)
        amp, i = get_peak_amplitudes(xpsd, dpsd)
        amplitudes.append(amp)
        tpos, y_or_z = outputThetaPosition(f, y_or_z)
        theta.append(tpos)
        if make_psd_plot:
            plt.loglog(freqs, xpsd)
            plt.plot(freqs[i], xpsd[i], "x")
    if plot_peaks:
        plt.figure()
        plt.plot(theta, amplitudes, 'o')
        plt.grid()
        plt.show(block = False)
    return

"""               # this is Fernando's plot             """
#plot_peaks2F(path)
""""""""""""""""""""""" THINGS HERE """""""""""""""""""""""

# this is Sumita's plot
def get_PSD_peak_parameters(file_list, use_theta = True):
    """ returns theta and ratio of [response at 2f] and [drive at f] """
    if use_theta:
        theta = []
        y_or_z = ""
    nx2 = []
    for f in file_list:
        if use_theta:
            tpos, y_or_z = outputThetaPosition(f, y_or_z)
            theta.append(tpos)
        freqs, xpsd, drivepsd = getdata(f) # Hz, V/sqrtHz, 1/sqrtHz
        freq_pos, twice_freq_pos = get_positions(xpsd, drivepsd) # bins
        nx2.append(conversion*(xpsd[twice_freq_pos])/(drivepsd[freq_pos])) # N
    if use_theta:
        return np.array(theta), np.array(nx2)
    else:
        return np.array(nx2) # Newtons

def getConstant(calibration_path):
    """ normalization to units of electrons """
    calibration_list = time_ordered_file_list(calibration_path)
    i = min(len(calibration_list), 20)
    nx2 = get_PSD_peak_parameters(calibration_list[:i], use_theta = False) # for one electron
    return np.average(nx2)# Newtons/electron

def plot_PSD_peaks(path, calib_path, last_plot = False):
    file_list = time_ordered_file_list(path)
    c = getConstant(calib_path) # Newtons/electron
    if debugging:
        print "c = ", c
    theta, nx2 = get_PSD_peak_parameters(file_list) # steps, Newtons
    nx2 = nx2/c # electrons
    plt.figure()
    plt.plot(theta, nx2, 'o')
    plt.xlabel('Steps in theta')
    plt.ylabel('Amplitude [electrons]')
    plt.title('PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show(block = last_plot)
    return

#plot_PSD_peaks(path, calib)

# now on to doing the area calibration thing
# integrating over basically the main peak
def get_area(f):
    w, x, d = getdata(f) # Hz, V/sqrtHz, 1/sqrtHz
    binF = w[1] - w[0] # Hz
    #x = x*conversion*sqrt(binF) # N
    gain, ACamp = getGainAndACamp(f) # unitless, V
    #constant = binF/(gain*ACamp) # Hz/V
    if debugging:
        fname = f[f.rfind('_')+1:f.rfind('.')]
        print ""
        print "DEBUGGING: get_area of ", fname
        print "           len(x) = ", len(x)
    i = np.argmax(d)
    if debugging:
        print "           i = ", i
        print ""
    return binF*sum(x[i-2:i+3])/(gain*ACamp) # sqrtHz

def peak_areas(path, c = 1, use_theta = False):
    a = [] # in units of measurement V/ V at 1 electron
    x = []
    y_or_z = ""
    if debugging:
        print ""
        print "DEBUGGING: peak_areas"
        if use_theta: print "           using theta"
    file_list = glob.glob(path + "/*.h5")
    if debugging:
        i = min(len(file_list), 20)
        file_list = file_list[:i]
    if len(file_list) == 1:
        return np.array([0]), np.array([get_area(file_list[0])])/c
    for f in file_list:
        if debugging:
            print "           reading file ", f[len(path):], "inside peak_areas"
        a.append(get_area(f))
        if use_theta:
            tpos, y_or_z = outputThetaPosition(f, y_or_z)
            x.append(tpos)
        else:
            x.append(int(f[f.rfind('_')+1:f.rfind('.')]))
    x, a = zip(*sorted(zip(x, a))) # sort by time
    if debugging:
        print "           peak_areas worked!"
        print ""
    return np.array(x), np.array(a)/c

def calibration_area(calib_path):
    x, a = peak_areas(calib_path)
    i = min(len(a), 20) # take first few files
    if debugging:
        print ""
        print "DEBUGGING: calibration_area"
        print "           i = ", i
        print ""
    one_electron = np.average(a[:i])
    return one_electron*num_electrons_in_sphere

def get_area_parameters(path, calib_path, use_theta = False):
    print "calibrating from ", calib_path
    c = calibration_area(calib_path)
    print "c = ", c
    print "finding areas for ", path
    x, a = peak_areas(path, c = c, use_theta = use_theta)
    return x, a # noise floor

def plot_areas(path, calib_path, use_theta = False, last_plot = False):
    x, a = get_area_parameters(path, calib_path, use_theta)
    print "Noise floor is at ", np.average(a), "fractions of an electron charge"
    print ""
    plt.figure()
    plt.loglog(x, a, 'o')
    if use_theta:
        plt.xlabel('Steps in theta')
    else:
        plt.xlabel('time [s]')
    plt.ylabel('peak area [electrons]')
    plt.title('Areas of PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show(block = last_plot)
    return


#plot_areas(path, calib, use_theta = True)
""""""""""""""""""""" Inputs """""""""""""""""""""
### calibration files
calib1 = "/data/20170622/bead4_15um_QWP/charge9"
calib2 = "/data/20170622/bead4_15um_QWP/arb_charge"

### this is where the noise files are pulled out
path1 = "/data/20170622/bead4_15um_QWP/reality_test2"
path2 = "/data/20170622/bead4_15um_QWP/reality_test3"
plot_areas(path1, calib1)
plot_areas(path2, calib2, last_plot = True)
