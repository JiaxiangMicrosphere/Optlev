import numpy as np
import time
from datetime import datetime
import os
import h5py
import scipy.signal as sp
import matplotlib.pyplot as plt
import glob
import matplotlib.mlab as mlab
import scipy.optimize as opt
from scipy.signal import find_peaks
import scipy.fft
from os.path import exists
from scipy import signal

electron_charge = -1.6e-19 # coulomb
pi = np.pi
countplot = 0 #number of the plot of te frequency comb
impulse_threshold = 1.5
# PXI1Slot3/ai0:3,PXI1Slot3/ai6:7,PXI1Slot3/ai16:21

#x channel
Amp_para=200.
HV_ch_x = 3
x_ch = 0

#y channel
Amp_para=100.
HV_ch_y = 9
y_ch = 1

#z channel
Amp_para=100.
HV_ch_z = 10
z_ch = 2

calibration1e = True
if not calibration1e :
    charge = -1*electron_charge# coulomb
else:
    charge = electron_charge

# important comment: The posititve X direction is choosen to be pointing away of the HV electrode and tongue. Empirically, for a sphere with
# negative charges (lots ot them), a positive DC voltage in the same electrode will make the sphere go towards negative X, therefore a negarive force
# which is nice because the force is Q*(-|e|) which is negative too, with Q = Capacitance*V > 0.
# For consistency, the calibration force with 1e must be negative and positive for a calibration with 1p.

# mass = 1*1e-12 # kg
#mass = 1.18*1e-13 # kg for 5um sphere
mass = 2.55*1e-14 # kg for 3um sphere

NFFT = 2**17

folder_cal_x=r"F:\data\20220721\3um_SiO2\2\1e_2\calibration_x\Dx-0.5_Dy1_Pz0.08_Dz1.7"
folder_cal_y=r"F:\data\20220721\3um_SiO2\2\1e_2\calibration_y\Dx-0.5_Dy0.5_Pz0.08_Dz1.65"
folder_cal_z=r"F:\data\20220721\3um_SiO2\2\1e_2\calibration_z\Dx-0.5_Dy1_Pz0.08_Dz1.65"

folder_sens = r"F:\data\20220721\3um_SiO2\2\momentum_cal_2\x\500MeV"
folder_sensor_noise = r"F:\data\20220721\2\no_sphere"

# electrodes_distance = 0.0033# old setting
electrodes_distance = 0.0365

# comsol_correction = 0.7
comsol_correction = 0.80  #new comsol result

### files

filelist_calibration_x = glob.glob(folder_cal_x + "/*.h5")
filelist_calibration_y = glob.glob(folder_cal_y + "/*.h5")
filelist_calibration_z = glob.glob(folder_cal_z + "/*.h5")
# filelist_calibration = filelist_calibration[0:2]
filelist_meas = glob.glob(folder_sens + "/*.h5")
filelist_sensor_noise = glob.glob(folder_sensor_noise + "/*.h5")

######## transfer functions
def apply_filters_fft(input_waveform, sampling_freq, band_list,f0):
  """ Function to filter a waveform by removing frequencies
      within the given bands using an FFT based filter

      Parameters:
        input_waveform - waveform to filter
        sampling_freq - Sampling frequency, Hz
        band_list - array of start/stop frequencies, the FFT of the waveform
                    is zeroed in the range between each start/stop freq

      Returns:
        filtered_waveform - time stream of filtered waveform
  """

  ## Assume real valued sequence and take fft of waveform
  g = np.fft.rfft(input_waveform)
  freqs  = np.fft.rfftfreq(len(input_waveform), 1/sampling_freq)
  ## loop over the filter bands and make a boolean array of frequencies to remove
  freqs_to_zero = freqs<0
  for current_band in band_list:
    freqs_to_zero = np.logical_or(freqs_to_zero, ( (freqs>=current_band[0]) & (freqs<=current_band[1]) ))
    if (current_band[0]>0 and current_band[1]<max(freqs)and max(abs(current_band-f0))<0.25*f0):
        g[np.where((freqs>=current_band[0]-0.1) & (freqs<=current_band[1]+0.1))]=(np.mean(g[np.where(freqs<=current_band[0])[0][-1]-2:np.where(freqs<=current_band[0])[0][-1]-1])
                                                                                    +np.mean(g[np.where(freqs>=current_band[1])[0][0]+1:np.where(freqs>=current_band[1])[0][0]+2]))/2
    elif (current_band[0] > 0 and current_band[1] < max(freqs) and min(abs(current_band - f0)) >= 0.25 * f0):
        g[np.where((freqs >= current_band[0] - 0.1) & (freqs <= current_band[1] + 0.1))] = (np.mean(g[np.where(freqs <= current_band[0])[0][-1] -5:np.where(freqs <= current_band[0])[0][-1] - 1])
                                                                                    + np.mean(g[np.where(freqs >= current_band[1])[0][0] + 1:np.where(freqs >= current_band[1])[0][0] + 5])) / 2
    elif (current_band[0] > 0 and current_band[1] < max(freqs) and max(abs(current_band - f0) >= 0.3 * f0) and min(abs(current_band - f0)) < 0.3 * f0):
        g[np.where((freqs >= current_band[0] - 1) & (freqs <= current_band[1] + 1))] = (np.mean(g[np.where(freqs <= current_band[0])[0][-1] -8:np.where(freqs <= current_band[0])[0][-1] - 3])
                                                                                    + np.mean(g[np.where(freqs >= current_band[1])[0][0] + 3:np.where(freqs >= current_band[1])[0][0] + 8]))/2
    else:
        g[np.where((freqs >= current_band[0]) & (freqs <= current_band[1]))]=0
  # g[freqs_to_zero] = 0 ## zero out noisy frequencies
  filtered_waveform =  np.fft.irfft(g) ## inverse transform back to time domain

  return filtered_waveform
def harmonic_transfer_function(f, f0, gamma0, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi * f0
    g0 = 2.*np.pi*gamma0
    a = A/(w0**2 - w**2 + 1j*w*g0)
    return np.real(a)

def harmonic_psd(f, f0, gamma0, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi * f0
    g0 = 2.*np.pi*gamma0
    a = (A**2)/((w0**2 - w**2)**2 + (w*g0)**2)
    return a

def harmonic(f, f0, g, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi*f0
    G = 2.*np.pi*g
    a1 = 1.*np.sqrt(  (w**2 - w0**2)**2 + (w*G)**2  )
    return 1.*A/a1
def log_harmonic(f,f0,g,A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi*f0
    G = 2.*np.pi*g
    a1 = np.log(1.*A/np.sqrt(  (w**2 - w0**2)**2 + (w*G)**2 ))
    return 1.*a1
#### CALIBRATION

def getdata(fname,num):
    print("Opening file: ", fname)
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    Fs = dset.attrs['Fsamp']
    pressure = dset.attrs['pressures'][0]
    #time = dset.attrs["Time"]

    dat = dat * 10. / (2 ** 15 - 1)
    if num==0:
        HV = dat[:, HV_ch_x] - np.mean(dat[:, HV_ch_x])
        x = dat[:, x_ch] - np.mean(dat[:, x_ch])
    if num==1:
        HV = dat[:, HV_ch_y] - np.mean(dat[:, HV_ch_y])
        x = dat[:, y_ch] - np.mean(dat[:, y_ch])
    if num==2:
        HV = dat[:, HV_ch_z] - np.mean(dat[:, HV_ch_z])
        x = dat[:, z_ch] - np.mean(dat[:, z_ch])

    return [Fs, x, HV,pressure]

def getnofield(nofieldlist,num):
    xpsd = 0
    for i in nofieldlist:
        data = getdata(i,num)
        Fs = data[0]
        x = data[1]
        # b, a = scipy.signal.iirnotch(40. / (Fs / 2), 50)
        # x = signal.filtfilt(b, a, x)
        # b, a = scipy.signal.iirnotch(60. / (Fs / 2), 50)
        # x = signal.filtfilt(b, a, x)
        # b, a = scipy.signal.iirnotch(14. / (Fs / 2), 50)
        # x = signal.filtfilt(b, a, x)
        # b, a = scipy.signal.iirnotch(68.5 / (Fs / 2), 50)
        # x = signal.filtfilt(b, a, x)
        # b, a = scipy.signal.iirnotch(29.2 / (Fs / 2), 50)
        # x = signal.filtfilt(b, a, x)
        # b, a = scipy.signal.iirnotch(9.5 / (Fs / 2), 10)
        # x = signal.filtfilt(b, a, x)
        psd, f = mlab.psd(x, Fs=Fs, NFFT=NFFT)
        xpsd = xpsd + psd
    xpsd = xpsd/len(nofieldlist)

    return [xpsd, f]

def getsensitivity(nofieldlist,num):
    xpsd = []
    j=0
    P=[]
    for ii in range(0,len(nofieldlist)):
        i=nofieldlist[ii]
        data = getdata(i,num)
        Fs = data[0]
        x = data[1]
        P=np.append(P,data[3])
        psd, f = mlab.psd(x, Fs=Fs, NFFT=NFFT)
        if j==0:
             xpsd = np.append(xpsd,np.abs(psd))
        else:
             xpsd = np.vstack((xpsd,np.abs(psd)))
        j=j+1
    return [xpsd, f,P]

def getforcesense(list_meas,list_calibration,num, electrodes_distance, comsol_correction):
    fpsd = []
    j = 0
    P = []
    from_daq_to_m, index, index2, f0, g = calibration_daqV_to_meters(list_calibration,num, electrodes_distance,  comsol_correction,False)
    print(from_daq_to_m, f0, g)
    for ii in range(0, len(list_meas)):
        i = list_meas[ii]
        data = getdata(i,num)
        Fs = data[0]
        x = data[1]

        P_0=time.ctime(os.path.getmtime(list_meas[0]))
        P_0 = datetime.strptime(str(P_0), "%a %b %d %H:%M:%S %Y")
        P_t=time.ctime(os.path.getmtime(i))
        P_t=datetime.strptime(str(P_t), "%a %b %d %H:%M:%S %Y")
        P_t=(P_t - P_0).seconds
        P = np.append(P, P_t)


        # P = np.append(P, data[3])
        psd, f = mlab.psd(x, Fs=Fs, NFFT=NFFT)
        w0 = 2. * pi * f0
        G = 2. * pi * g
        w = 2. * pi * f
        h = (w ** 2 - w0 ** 2) ** 2 + (w * G) ** 2
        fpsd0 = abs(from_daq_to_m) * mass * ((psd) ** 0.5) * np.sqrt(h)
        if j == 0:
            fpsd = np.append(fpsd, fpsd0)
        else:
            fpsd = np.vstack((fpsd, fpsd0))
        j = j + 1
    return [fpsd, f, P]


#[indexes, indexes2, psdhv, psdhv2]
def get_freqs_comb_calibration(list_calibration,num, plot):

    HVlist = []
    for i in list_calibration:
       data  = getdata(i,num)
       Fs = data[0]
       HVlist.append(data[2])

    psdhv = 0.
    psdhv2 = 0.
    for j in HVlist:
        # 1f peaks
        psd, f = mlab.psd(j, Fs=Fs, NFFT=NFFT)
        psdhv = psdhv + psd
        # 2f peaks (for induced dipole)
        m = j**2 - np.mean(j**2)
        psd2, f = mlab.psd(m, Fs=Fs, NFFT=NFFT)
        psdhv2 = psdhv2 + psd2

    psdhv = psdhv/len(HVlist)
    psdhv2 = psdhv2/len(HVlist)

    indexes = np.where(psdhv > 0.50*max(psdhv))[0]
    indexes2 = np.where(psdhv2 > 0.25*max(psdhv2))[0]

    # indexes = np.where(psdhv > 9e-6)[0]
    # indexes2 = np.where(psdhv2 > 1.6e-7)[0]


    indexes2 = np.array(indexes2)
    if num==0:
        indexes= indexes[15:60]
        indexes2= indexes2[15:60]
    if num == 1:
        indexes = indexes[15:50]
        indexes2 = indexes2[15:50]
    if num==2:
        indexes= indexes[60:150]
        indexes2= indexes2[60:150]
    if plot:
        plt.figure()
        plt.loglog(f, np.sqrt(psdhv))
        plt.loglog(f[indexes], np.sqrt(psdhv[indexes]), "o")
        plt.loglog(f, np.sqrt(psdhv2))
        plt.loglog(f[indexes2], np.sqrt(psdhv2[indexes2]), "s")
        plt.xlabel("Freq[Hz]")
        plt.ylabel(r"PSD[$V/\sqrt{Hz}$]")
        plt.legend()
        plt.show()

    #indexes, indexes2 = forbiden_freqs(f, indexes, indexes2, forbiden_list)

    # this is the voltage output of the synth into the electrodes
    voltage = np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes]))
    print ("synth voltage RMS during electrode calibration in V = ", voltage)
    electrode_voltage = Amp_para*voltage

    electrode_voltage_1peak = Amp_para*np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes[0:2]]))
    print("electrode voltage first peak RMS during electrode calibration in V = ", electrode_voltage_1peak)

    electrode_voltage = Amp_para*np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes]))
    print("electrode voltage RMS during electrode calibration in V = ", electrode_voltage)

    if False:
        xpsd = 0
        plt.figure()
        for i in list_calibration:
            x = getdata(i)[1]
            f, psd = sp.csd(x, x, Fs, nperseg=NFFT, scaling="spectrum")
            xpsd = xpsd + psd/(np.sqrt(2.)*Fs/NFFT)
        xpsd = xpsd/len(list_calibration)
        plt.loglog(f, np.sqrt(xpsd), label = "1e")
        plt.loglog(f[indexes], np.sqrt(xpsd[indexes]), "s",)
        plt.xlabel("freq [Hz]")
        plt.ylabel("V/sqrt(Hz)")
        plt.legend()

    return [indexes, indexes2, psdhv, psdhv2]
#[fieldpsd, xpsdhv, f, index, df] in V^2 / Hz
def HV_field_and_X_psd(list_calibration,num, electrodes_distance, comsol_correction):
    global countplot
    if countplot==0:
        data = get_freqs_comb_calibration(list_calibration,num, True)
        countplot =1
    else:
         data = get_freqs_comb_calibration(list_calibration,num, False)

    # field psd
    index = data[0]
    index2 = data[1]
    vpsd = data[2]
    vpsd = vpsd[index]
    fieldpsd = vpsd*(Amp_para*comsol_correction/electrodes_distance)**2 ## remember that vpsd is in V^2/Hz that is why the ()**2

    # X psd in volts units (daq)
    Xlist = []
    for i in list_calibration:
       datax  = getdata(i,num)
       Fs = datax[0]
       Xlist.append(datax[1])

    xpsdhv = 0.
    for j in Xlist:
        psd, f = mlab.psd(j, Fs=Fs, NFFT=NFFT)
        xpsdhv = xpsdhv + psd

    df = f[1]-f[0]

    xpsdhv = xpsdhv/len(Xlist)

    xpsdhv = xpsdhv[index]
    f = f[index]

    return [fieldpsd, xpsdhv, f, index, df, index2]

def calibration_daqV_to_meters(list_calibration,num, electrodes_distance, comsol_correction,plot):

    data = HV_field_and_X_psd(list_calibration,num, electrodes_distance, comsol_correction)
    fieldpsd = data[0]
    xpsdhv = data[1]
    f = data[2]
    df = data[4]
    index = data[3]
    index2 = data[5]

    p0 = [40, 1, 1]
    popt, pcov = opt.curve_fit(harmonic, f, xpsdhv**0.5, p0 = p0, sigma = xpsdhv**0.5)

    f0 = np.abs(popt[0])
    g = np.abs(popt[1])
    print('gamma=',g)
    h = ((fieldpsd*df)**0.5)*(charge/mass)

    displacement_m = harmonic(f, f0, g, h)

    from_daq_to_m = np.mean( displacement_m / ((xpsdhv*df)**0.5) )

    # from_daq_to_m = -np.abs(from_daq_to_m) # think about that, this number only depends on the optics and wiring of the daq, has nothing to do with charge.
                                                         # the "issue" is that the charge changes sign above, but the xpsdhv is always positive regardless if there is 1e ou 1p at the calibration files.
    # important thing that sets the sign is the following, if the sphere has 1e and there is a positive voltage in the biased electrode, we know the force is
    # negative (coordinate going from biased electroded to grounded one). Does the choices makes this happen?

    if (True&plot):
        plt.figure()
        freqplot = np.linspace(10, 200, 1000)
        plt.loglog(f, xpsdhv**0.5, "o")
        plt.loglog(freqplot, harmonic(freqplot, *popt))
        print(popt)
        plt.xlabel("Freq[Hz]")
        plt.ylabel(r"PSD[$V/\sqrt{Hz}$]")
        plt.title("CHECK THIS FIT for the calibration")
        plt.legend()
        plt.show()

    if False:
        #plt.figure()
        #plt.loglog(f, xpsdhv**0.5)
        #plt.loglog(f, harmonic(f, *popt))
        #plt.loglog(f, 1e7*displacement_m)
        #plt.figure()
        #plt.loglog(f, displacement_m/xpsdhv**0.5)
        #plt.loglog(f, from_daq_to_m*xpsdhv**0.5, ".")
        #plt.loglog(displacement_m, harmonic(f, *popt), ".")

        nofxpsd = getnofield(filelist_meas)

        #plt.figure()
        #plt.loglog(nofxpsd[1], from_daq_to_m*(nofxpsd[0]**0.5))
        #plt.figure()
        #plt.loglog(f, fieldpsd**0.5)

        force_s = mass*from_daq_to_m*((nofxpsd[0])**0.5)*np.sqrt(  ((2*np.pi*nofxpsd[1])**2 - (2*np.pi*f0)**2)**2 + (2*np.pi*nofxpsd[1]*2*np.pi*g)**2  )
        plt.figure()
        plt.loglog(nofxpsd[1], np.abs(force_s))
        plt.show()

    #test force 1f with the calibration

    # testdata  = getdata(filelist_calibration[0])
    # Fs = testdata[0]
    # mcal = testdata[1]*from_daq_to_m
    # volt = testdata[2]
    #
    # f, c1 = sp.csd(volt, mcal, Fs, nperseg=NFFT, scaling = "spectrum")
    # f, c2 = sp.csd(volt, volt, Fs, nperseg=NFFT, scaling = "spectrum")
    #
    # fo = c1/(c2**0.5)
    # fo = fo*((2*np.pi)**2)*( f0**2 - f**2 + 1j*f*g  ) ## the -1j is due to the complex conjugate
    # fo = np.real(fo)
    # fo = mass*fo
    #
    # #print (np.sum(fo[index]**2)**0.5)
    #
    # sign = np.mean((fo[index]/np.abs(fo[index])))
    # print(sign)
    # print (sign*np.sum( (np.sum(fo[index] ** 2) ** 0.5) ))
    # print (charge*np.sum(fieldpsd*df)**0.5)
    #
    # plt.figure()
    # plt.loglog(f, np.abs(fo)) # the abs here is only because the force can be negative and this is a loglog plot. Not necessary otherwise
    # plt.plot(f[index], np.abs(fo[index]), ".")
    # plt.show()
    #quit()

    return [from_daq_to_m, index, index2, f0, g]

def force_psd(list_meas, list_calibration,num, electrodes_distance, comsol_correction):

    xpsd, f = getnofield(list_meas,num)

    from_daq_to_m, index, index2, f0, g = calibration_daqV_to_meters(list_calibration,num, electrodes_distance, comsol_correction,False)

    w0 = 2.*pi*f0
    G = 2.*pi*g
    w = 2.*pi*f

    h = (w**2 - w0**2)**2 + (w*G)**2

    fpsd = from_daq_to_m * mass * ((xpsd) ** 0.5) * np.sqrt(h)

    # plt.figure()
    # plt.loglog(f, np.abs(fpsd))
    #
    # plt.xlabel("$f$ (Hz)", fontsize=15)
    # plt.ylabel(r"$F$ (N)", fontsize=15)
    #
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout(pad=0)
    # plt.legend()
    return [f, np.abs(fpsd)]
def force_psd_2(list_meas,num,from_daq_to_m, electrodes_distance, comsol_correction,f0,g): ##only after finishing the calibration can this be used
    xpsd, f = getnofield(list_meas,num)
    w0 = 2. * pi * f0
    G = 2. * pi * g
    w = 2. * pi * f
    h = (w ** 2 - w0 ** 2) ** 2 + (w * G) ** 2
    fpsd = from_daq_to_m * mass * ((xpsd) ** 0.5) * np.sqrt(h)
    return [f, np.abs(fpsd)]
def force_estimator_time(v_time_stream, v_to_m, f0, Gamma, Fs):

    # get fft of displacement

    displacement = v_time_stream*v_to_m

    time = np.linspace(0, (len(displacement))/Fs, len(displacement))

    xfft = scipy.fft.rfft(displacement)
    n = displacement.size
    freq = np.fft.rfftfreq(n, d = 1./Fs)

    #test to check the fft and inverse fft
    #plt.figure()
    #plt.plot(displacement)
    #new_dis = scipy.fft.irfft(xfft)
    #print (np.max(np.abs(np.imag(new_dis))))
    #print(np.min(np.abs(np.real(new_dis))))
    #plt.plot(new_dis)

    # do mass * 1/harmonic_transfer_function

    Hinv = 1./harmonic_transfer_function(freq, f0, Gamma, 1)

    force_fft = mass*Hinv*xfft

    # inverse fft

    force = scipy.fft.irfft(force_fft)
    bandlist=[[0.,20],[35.5,36.1],[59.9,60.1],[40.02,40.43],[100,max(freq)]]
    # filter
    y3=apply_filters_fft(force,Fs,bandlist,f0)


    # wn1 = 0.5*f0/(Fs/2)
    # wn2 = 2*f0/(Fs/2)
    # b, a = signal.butter(2, Wn = [wn1, wn2], btype = "bandpass")
    # y = signal.filtfilt(b, a, force)
    #
    # bn, an = scipy.signal.iirnotch(60./(Fs/2), 20)
    # # bn2, an2 = scipy.signal.iirnotch(62.8 / (Fs / 2), 20)
    # # bn3, an3 = scipy.signal.iirnotch(49.4 / (Fs / 2), 80)
    # # bn3, an3 = scipy.signal.iirnotch(53.8/ (Fs / 2), 20)
    # # bn4, an4 = scipy.signal.iirnotch(49.4 / (Fs / 2), 20)
    # bn5, an5 = scipy.signal.iirnotch(40.1 / (Fs / 2), 20)
    # y2 = signal.filtfilt(bn, an, y)
    # # y3 = signal.filtfilt(bn2, an2, y2)
    # # y3 = signal.filtfilt(bn3, an3, y3)
    # # y3 = signal.filtfilt(bn4, an4, y3)
    # y3 = signal.filtfilt(bn5, an5, y2)

    # plt.figure()
    # plt.plot(time,y)
    # plt.plot(time[10000:110000],y2[10000:110000])
    # plt.plot(time[30000:-30000],y3[30000:-30000])
    # plt.xlabel("$t$ (s)", fontsize=15)
    # plt.ylabel(r"$F$ (N)", fontsize=15)
    #print ( ((np.mean(y3[80000:120000]**2)**0.5)**2/(50))**0.5 )

    #test force psd
    # psdforce, freqpsd = mlab.psd(force, Fs=Fs, NFFT=NFFT)
    #psdfy, freqpsd = mlab.psd(y, Fs=Fs, NFFT=NFFT)
    # psdfy3, freqpsd = mlab.psd(y3, Fs=Fs, NFFT=NFFT)
    #plt.figure()
    #plt.loglog(freqpsd, psdforce**0.5)
    #plt.loglog(freqpsd, psdfy**0.5)
    # plt.loglog(freqpsd, psdfy3 ** 0.5)
    # print ( ((np.mean(y3[80000:120000]**2)**0.5)**2/(50))**0.5 )

    return [y3, time]

def displacement_estimator_time(v_time_stream, v_to_m, f0, Gamma, Fs):

    displacement = v_time_stream*v_to_m

    time = np.linspace(0, (len(displacement))/Fs, len(displacement))

    xfft = scipy.fft.rfft(displacement)
    n = displacement.size
    freq = np.fft.rfftfreq(n, d = 1./Fs)
    # get fft of displacement
    bandlist = [[0., 20], [35.5, 36.1], [59.9, 60.1], [40.02, 40.43], [100, max(freq)]]
    # filter
    v_filt = apply_filters_fft(v_time_stream, Fs, bandlist,f0)
    displacement = v_filt * v_to_m

    return [displacement, time]
# functions from the rudimentary time-stream fitting algorithm

class underdamped_oscillator:
    def __init__(self, time, amp, omega, gamma, delta, delay, Fs):
        self.time = time
        self.amp = amp
        self.omega = omega
        self.gamma = gamma
        self.delta = delta
        self.delay = delay
        self.Fs = Fs

    def f(self):
        array = np.zeros(len(self.time))
        trimmedtime = (self.time * self.Fs).astype(int)[self.delay:]
        array[self.delay:] = self.amp * np.exp(-self.gamma / 2 * (trimmedtime - self.delay) / self.Fs) * np.sin(
            self.omega * (trimmedtime - self.delay) / self.Fs + self.delta)
        return array

    def template(self, length_template):
        t = np.array(range(0, length_template))
        array = self.amp * np.exp(-self.gamma / 2 * t / self.Fs) * np.sin(self.omega * t / self.Fs + self.delta)
        # for i in range(len(array)):
        #    if t[i]/self.Fs>4*np.log(20)/self.gamma:
        #        array[i]=0
        return array, t  # trim the template to a given length and then leave zeros.
def  momentum_impulse_file(file_meas, list_calibration, list_noise_files,electrodes_distance, comsol_correction, filter=False, matched=False):

    from_daq_to_m, _, _, f0, g = calibration_daqV_to_meters(list_calibration, electrodes_distance,
                                                            comsol_correction,True)
    vars = np.array([from_daq_to_m, f0, g])
    axes=1
    np.save(folder_cal + '\\' + str(axes) + 'calibration_data.npy', vars)
    oscillator_params = [1, np.sqrt((f0 * 2 * np.pi) ** 2 - g ** 2 / 4), g]

    # position signal
    Fs, x, HV_saved,_ = getdata(file_meas)

    displacement = x * from_daq_to_m

    xfft = np.fft.rfft(displacement, n=NFFT)
    n = NFFT
    print(n)
    freq1 = np.fft.rfftfreq(n, d=1. / Fs)

    time = np.array(range(len(displacement)))
    delta = 0
    underdamped = underdamped_oscillator(time, 1, oscillator_params[1], oscillator_params[2], delta, 0,
                                         Fs)  # create a template for our oscillator in the time domain, for x
    template, _ = underdamped.template(len(displacement))
    templatefft = np.fft.rfft(template,n=NFFT)  # scipy or numpy fft is exactly the same, this is for using the unit impulse as our template
    H = harmonic_transfer_function(freq1, f0, g, 1)
    forcefft_template = templatefft * H * mass
    templatefft = templatefft / np.max(np.fft.irfft(
        forcefft_template))  # normalize the template. This is probably the point in which the estimate goes off from the theoretical value

    # apply filters or matched filter and convolute, extract impulse information
    if matched:
        if exists(folder_sensor_noise + '\\' + str(
                axes) + 'noisepsd.npy') and not change_NFFT:  # same thing as with the calibration. We save time by not computing everything multiple times.
            xnoise_psd = np.load(noise_folders[axes - 1]+ '\\'  + str(axes) + 'noisepsd.npy')
        else:
            # noise psd
            print("Processing noise...")
            counter = 0
            for i in list_noise_files:  # this loop averages all the noise psds with no events
                counter = counter + 1
                Fs, xn, HV,_ = getdata(i)
                displacementn = xn * from_daq_to_m

                psd, f = mlab.psd(displacementn, Fs=Fs, NFFT=NFFT)
                if counter == 1:
                    xnoise_psd = np.abs(psd)
                else:
                    xnoise_psd = xnoise_psd + np.abs(psd)  # if we want to fit with the force

            xnoise_psd = xnoise_psd / counter
            # badfreq = np.where((f>250)|(f<10))[0] # if this line and the next are uncommented we cut off the frequencies in the argument of np.where
            # xnoise_psd[badfreq]=1e26

            np.save(folder_sensor_noise+ '\\' + str(axes) + 'noisepsd.npy',
                    xnoise_psd)  # we store our variable in case that we are just playing with different
            # measurement files. This way, we save ~15s of loading files per run. It is not a significant difference but it could add up if we
            # wanted to process many runs at once

            change_NFFT = False
        # "optimum" filter
        B = np.real(np.sum(np.divide(np.multiply(np.conj(templatefft), templatefft), xnoise_psd)))
        A = np.multiply(np.conjugate(templatefft), xfft) / xnoise_psd  # this line works fitting force
    else:
        B = np.real(np.sum(np.multiply(np.conj(templatefft), templatefft)))
        A = np.multiply(np.conj(templatefft), xfft)

    P = np.real(np.fft.irfft(
        A) / B * NFFT)  # normalization factor of NFFT needed because of the difference between a DFT formula and the Golwala formula

    # diagnosis plot in frequency domain
    global filenum
    fignoise = plt.figure(filenum)
    axnoise = fignoise.add_axes([0.15, 0.15, 0.70, 0.70])
    if matched:
        axnoise.loglog(freq1, np.abs(np.conj(templatefft) / np.sqrt(xnoise_psd)) / np.sum(
            np.abs(np.conj(templatefft) / np.sqrt(xnoise_psd))), label="Filter")
        axnoise.loglog(freq1, np.sqrt(xnoise_psd), label='Noise')
    # axnoise.loglog(freq1,np.abs(A/B),label='A')
    # axnoise.loglog(freq1, np.abs(xfft), label='Measurement', alpha=0.7)
    # axnoise.loglog(freq1, np.abs(templatefft), label='Template')

    if matched:
        axnoise.set_title("Matched filter and filtered data, file " + str(filenum))
    else:
        axnoise.set_title("Template for convolution and convoluted data" + str(filenum))
    axnoise.set_xlim([0.1, 5000])
    axnoise.set_ylabel('PSD [m/$\sqrt{Hz}]$')
    axnoise.set_xlabel('f [Hz]')
    fignoise.legend(loc='right')

    P = (P / Fs / 5.36e-22)[0:len(
        x)]  # with this line we convert force (N) to impulse (MeV/c)  (FΔt=Δp). The slicing is important in case we are using an excess of FFT coefficients
    if matched:
        golwala_error = np.sqrt(
            (np.sum(np.abs(templatefft) ** 2 / np.sqrt(xnoise_psd)) * 2 * len(displacement) / Fs) ** (
                -1)) / Fs / 5.36e-22
        # golwala_error = np.sqrt((np.sum(np.abs(templatefft)**2/np.sqrt(xnoise_psd)))**(-1))/Fs/5.36e-22
    # diagnosis plot in time domain

    fig_sampleimpulse = plt.figure(filenum + 600)  # do this to avoid overwriting figures
    axs = fig_sampleimpulse.add_axes([0.15, 0.15, 0.70, 0.70])
    axs.plot(displacement[0:len(x)], label='displacement', color='r', alpha=0.7, zorder=0)
    # axs.set_xlim([2500,10500])
    some_periods = int(5 / f0 * Fs)
    event_indexes_x, _ = signal.find_peaks(abs(P), height=impulse_threshold * np.std(P), distance=some_periods)
    axs1 = axs.twinx()
    axs1.scatter(event_indexes_x, P[event_indexes_x])
    axs1.set_ylim([-np.max(np.abs(P)), np.max(np.abs(P))])
    axs1.plot(P, label='extracted impulse', color='b', alpha=1, zorder=0)
    axs1.plot(HV_saved[0:len(x)] / 2 * np.max(P), label='HV', color='green')
    axs1.set_ylabel("Extracted impulse [MeV/c]")
    axs.set_ylim([-np.max(np.abs(displacement)), np.max(np.abs(displacement))])
    axs.set_ylabel("Displacement [m]")
    axs.legend(loc="upper left")
    axs1.legend(loc='upper right')
    axs1.set_title("File " + str(filenum))
    fig_sampleimpulse.savefig("Filtered_signal_" + str(filenum) + ".png", dpi=400)
    filenum = filenum + 1
    if matched:
        return P, golwala_error, some_periods  # this impulse should be in MeV/c. some_periods is the number of samples we must have between peaks.
    else:
        return P, some_periods  # this impulse should be in MeV/c


hf = h5py.File('momentum_calibrationx_500MeV_4.h5', 'w')
#force_psd(filelist_meas, filelist_calibration, electrodes_distance, comsol_correction)
#force_psd(filelist_calibration, filelist_calibration, electrodes_distance, comsol_correction)
from_daq_to_m=np.zeros(3)
f0=np.zeros(3)
g=np.zeros(3)
filelist_calibration=[filelist_calibration_x,filelist_calibration_y,filelist_calibration_z]
from_daq_to_m[0], index, index2, f0[0], g[0] = calibration_daqV_to_meters(filelist_calibration_x,0, electrodes_distance, comsol_correction,True)
from_daq_to_m[1], index, index2, f0[1], g[1] = calibration_daqV_to_meters(filelist_calibration_y,1, electrodes_distance, comsol_correction,True)
from_daq_to_m[2], index, index2, f0[2], g[2] = calibration_daqV_to_meters(filelist_calibration_z,2, electrodes_distance, comsol_correction,True)
hf.create_dataset('from_daq_to_m', data=from_daq_to_m)
hf.create_dataset('f0', data=f0)
hf.create_dataset('Fs', data=10000)
hf.create_dataset('g', data=g)
print ("x,y,z resonance frequency in calibration is", f0[0],f0[1],f0[2])
print ("x,y,z from_daq_to_m is",from_daq_to_m[0],from_daq_to_m[1],from_daq_to_m[2])
for i in [0,1,2]:
    Fs, v_time_stream, HV, P= getdata(filelist_meas[4],i)
    if i==0:
        hf.create_dataset('x', data=v_time_stream)
    if i==1:
        hf.create_dataset('y', data=v_time_stream)
    if i==2:
        hf.create_dataset('z', data=v_time_stream)
        HV_r=200*HV
        hf.create_dataset('HV', data=HV_r)
    print(v_time_stream, from_daq_to_m, f0, g)
    [x_disp,t_disp]=displacement_estimator_time(v_time_stream, from_daq_to_m[i], f0[i], g[i], Fs)
    plt.figure()
    plt.plot(t_disp[30000:-30000], x_disp[30000:-30000])
    plt.xlabel("$t$ (s)", fontsize=15)
    plt.ylabel(r"$X$ (m)", fontsize=15)
    plt.show()

    [Force,t_force]=force_estimator_time(v_time_stream, from_daq_to_m[i], f0[i], g[i], Fs)
    plt.figure()
    plt.plot(t_force[30000:-30000], Force[30000:-30000])
    plt.xlabel("$t$ (s)", fontsize=15)
    plt.ylabel(r"$F$ (N)", fontsize=15)
    plt.show()

    #Get displacement psd
    vpsd,f,P_0=getsensitivity(filelist_meas,i)
    if (vpsd.ndim>1):
        vpsd=np.mean(vpsd, axis=0)
    else:
        vpsd=vpsd
    plt.figure()
    plt.loglog(f,vpsd**0.5*np.abs(from_daq_to_m[i])) #from_daq_to_m sometime is negative
    plt.ylabel("displacement [m/$\sqrt{Hz}$]", {'size': 20})
    # plt.xlabel("Pressure [mbar]", {'size': 20})
    plt.xlabel("f [Hz]", {'size': 20})
    # plt.xticks(range(0,len(P),50),rotation ='vertical')
    plt.legend()
    plt.show()
    peaks, properties = find_peaks(vpsd, prominence=(1e-4,1e-2),width=(0,5))
    min_freq, max_freq = 10, 200 ## reduce range to just get lowest noise part
    arr = np.zeros(len(f));
    # Copying all elements of one array into another
    for j in range(0, len(arr)):
        arr[j] = f[j];
    for j in np.arange(int(len(properties["widths"]))):
        if abs(f[peaks][j]-f0[i])<5:
            properties["widths"][j]=0.1
        arr[(arr>f[peaks][j]-1/2*properties["widths"][j])&(arr<f[peaks][j]+1/2*properties["widths"][j])]=1
    arr[arr!=1]=0
    arr=arr.astype(np.bool)
    bad_freqs = arr|(f < min_freq) | (f > max_freq)
    bad_starts = np.argwhere(bad_freqs & (~np.roll(bad_freqs,1)))
    bad_ends = np.argwhere(bad_freqs & (~np.roll(bad_freqs,-1)))
    bad_starts = np.insert(bad_starts,0,0) #insert zero at the beginning of the start array
    bad_ends = np.append(bad_ends,-1) # add -1 to the end of the final block of frequencies
    filter_list = np.vstack( (f[bad_starts], f[bad_ends]) ).T # make n x 2 array of bad intervals to pass to apply_filters_fft
    print("Frequency ranges to remove:")
    print(filter_list)
    v_time=scipy.fft.irfft(vpsd)
    v_time=apply_filters_fft(v_time, Fs, filter_list,f0[i])
    vpsd=scipy.fft.rfft(v_time)
    xpsd=vpsd*(from_daq_to_m[i]**2)
    f1=np.where(f>14)[0][0]
    f2=np.where(f<17)[0][-1]
    plt.figure()
    plt.loglog(f[np.where(f>min_freq)[0][0]:np.where(f<max_freq)[0][-1]],vpsd[np.where(f>min_freq)[0][0]:np.where(f<max_freq)[0][-1]]**0.5*np.abs(from_daq_to_m[i])) #from_daq_to_m sometime is negative
    plt.ylabel("Displacement [m/$\sqrt{Hz}$]", {'size': 20})
    # plt.xlabel("Pressure [mbar]", {'size': 20})
    plt.xlabel("Frequency [Hz]", {'size': 20})
    # plt.xticks(range(0,len(P),50),rotation ='vertical')
    plt.legend()
    print("noise is:",np.mean(vpsd[f1:f2]**0.5*np.abs(from_daq_to_m[i])),"m")

    #force sensitivity with time
    x,f,P=getforcesense(filelist_meas,filelist_calibration[i], i,electrodes_distance, comsol_correction)
    # plt.figure()
    S_base=[]
    for j in range(0,len(filelist_meas)):
        f_1 = np.where(f>= 10.)[0][0]
        f_2= np.where(f<= 25.)[0][-1]
        # plt.loglog(f[100:2000], np.sqrt(x[i][100:2000]),label=filelist_meas[i][-9:-3])
        # plt.loglog(f[f_1:f_2], x[i][f_1:f_2], label=P[i])
        # plt.loglog(f[90:150], x[i][90:150], label=P[i])
        # plt.ylim(1e-19, 1e-14)
        # plt.ylabel("F [N/$\sqrt{Hz}$]", {'size': 20})
        # plt.xlabel("Frequency [Hz]", {'size': 20})
        # plt.legend()
        if np.mean(x[j][f_1:f_2])<1e-16:
            S_base=np.append(S_base,np.mean(x[j][f_1:f_2]))
        else:
            P=np.delete(P,j)
    # plt.show()
    # print(P[-1])
    plt.figure()
    # plt.loglog(P,S_base, "o")
    plt.scatter(P,S_base)
    plt.ylabel("F [N/$\sqrt{Hz}$]", {'size': 20})
    # plt.xlabel("Pressure [mbar]", {'size': 20})
    plt.xlabel("time [s]", {'size': 20})
    # plt.xticks(range(0,len(P),50),rotation ='vertical')
    plt.legend()
    plt.show()

    #fit the PSD
    Fs=10000
    [displacement,f]=getnofield(filelist_meas,i)
    p0 = [31., 0.0001, 1e-6]
    # p0=[38.52708253365734, 0.00016204920408411055, 2.9127599445516284e-06]
    f1=np.where(f>min_freq)[0][0]
    f2=np.where(f<max_freq*0.25)[0][-1]

    # popt, pcov = opt.curve_fit(harmonic, f[f1:f2], displacement[f1:f2]**0.5-np.mean(displacement[f1:f2]**0.5), p0 = p0) #unfiltered
    popt, pcov = opt.curve_fit(log_harmonic, f[f1:f2],np.log(xpsd[f1:f2]**0.5), p0 = p0) #filtered ,bounds=([0,0,0],[40,0.01,5e-6])
    f_dis = np.abs(popt[0])
    g_dis = np.abs(popt[1])
    A=np.abs(popt[2])
    chi_square=np.sum((xpsd[f1:f2].real**0.5-harmonic(f[f1:f2],f_dis,g_dis,A))**2/(xpsd[f1:f2].real**0.5))/len(f[f1:f2])
    print("chi square for the fitting is:",chi_square)
    freq_a,Force_a=force_psd_2(filelist_meas,i, from_daq_to_m[i], electrodes_distance, comsol_correction,f_dis,g_dis)
    print ("Resonance frequency in meas is", f0)
    print ("damping constant is", g)
    plt.figure()
    plt.loglog(freq_a, Force_a,label="fit from displacement")
    plt.xlabel("Frequency [Hz]",{'size':20})
    plt.ylabel("Force [$N/\sqrt{Hz}$]",{'size':20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0)
    plt.legend()
    plt.show()

    freq_b,Force_b=force_psd(filelist_meas, filelist_calibration[i],i, electrodes_distance, comsol_correction)
    fa1=np.where(f>14)[0][0]
    fa2=np.where(f<17)[0][-1]
    print("Force noise is:",np.mean(Force_b[fa1:fa2]),"N")
    # Force_psd, f = mlab.psd(Force , Fs=Fs, NFFT=NFFT)
    plt.figure()
    # plt.loglog(f, Force_psd**0.5)
    plt.loglog(freq_b, Force_b,label="fit from calibration")
    plt.xlabel("Frequency [Hz]",{'size':20})
    plt.ylabel("Force [$N/\sqrt{Hz}$]",{'size':20})
    plt.xlim(1, 500)
    plt.ylim(0.5e-19, 2e-17)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=0)
    plt.legend()
    plt.show()
    [displacement,f]=getnofield(filelist_meas,i)
    # p0 = [30., 1, 1]
    # # popt, pcov = opt.curve_fit(harmonic, f[f1:f2], displacement[f1:f2]**0.5, p0 = p0)
    # #
    # # f0 = np.abs(popt[0])
    # # g = np.abs(popt[1])
    # # A=np.abs(popt[2])

    #verify the fitting
    print([f_dis,g_dis,A])
    xfit=harmonic(f,f_dis,g_dis,A)
    # xfit=harmonic(f,f0,0.02,2e-6)
    print("frequency is:",f_dis,"damping is:",g_dis)
    disp_fig = plt.figure(figsize = (12,4))
    plt.figure(disp_fig.number)
    plt.subplot(1, 2, 1)
    # plt.loglog(f, displacement**0.5)
    plt.loglog(f[f1:], displacement[f1:]**0.5)
    plt.loglog(f[f1:], (xfit[f1:])/np.abs(from_daq_to_m[i]))
    plt.loglog(f[f1:],xpsd[f1:]**0.5/np.abs(from_daq_to_m[i]))
    plt.xlabel("Frequency [Hz]",{'size':20})
    plt.ylabel("Voltage [$V/\sqrt{Hz}$]",{'size':20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0)
    plt.legend()
    #Fs, v_time_stream, HV = getdata(filelist_meas[0])
    # momentum_estimator_time(v_time_stream, from_daq_to_m, f0, g, Fs)
    plt.subplot(1, 2, 2)
    plt.loglog(f[f1:], displacement[f1:]**0.5*np.abs(from_daq_to_m[i]))
    plt.loglog(f[f1:], xfit[f1:])
    # plt.loglog(f[f1:],xpsd[f1:]**0.5)
    plt.xlabel("Frequency [Hz]",{'size':20})
    plt.ylabel("Displacement [$m/\sqrt{Hz}$]",{'size':20})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0)
    plt.legend()
    plt.show()
    filenum=0
    # Px_ind,  some_periods = momentum_impulse_file(filelist_meas[0], filelist_calibration[i],i, filelist_sensor_noise,electrodes_distance, comsol_correction, filter=True, matched=True)
    # plt.show()