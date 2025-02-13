import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

single_channel = False
VEOM_h5 = True
measuring_with_xyz = True

scope = True # gets the correct FS

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


savetxt = True

#path = r"C:\data\20180215\bead4_VAT_POL_NS\EOM_4e-2mbar_meas9"
#path = r"C:\data\20180702\bead2_SiO2_15um_POL_NS\rotation3"
#path = r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_new_turbo_3_V+10"
path = r"C:\data\20180925\bead1_SiO2_15um_POL_NS\rotation2"

file_list = glob.glob(path+"\*.h5")

file_list = list_file_time_order(file_list)

file_list = file_list[-5:]

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**14

if single_channel:
    a = 0
else:
    a = bu.xi_old

if measuring_with_xyz:
    a = -1


def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
                Press = dset.attrs['temps'][0]
                Volt = 0.0
                if VEOM_h5:
                    Volt = dset.attrs['EOM_voltage']
                if scope:
                    Fs = dset.attrs['FS_scope']
                		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                Time = dset.attrs["Time"]
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, a]-numpy.mean(dat[:, a]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, 0, 0, 0, 0, xpsd_old, Press, Volt, Time]


freq = getdata(file_list[0])[0]
time0 = getdata(file_list[0])[8]
N = 1
cmap = bu.get_color_map(len(file_list)/N)
for idx,c in zip(range(len(file_list)/N), cmap):
        i = file_list[idx]
        aux_press = str("mbar")
        aa = getdata(i)
        aux = str(aa[6])
        aux_angle = str("_angle=")
        aux2 = i[i.rfind('_')+1:i.rfind('.h5')]
        v = ""
        if VEOM_h5:
            v = ", v="+str("%.1f" % aa[7])
        tot_psd = 0
        for j in range(N):
           cpsd = getdata(file_list[idx*N + j])[5]
           tot_psd += cpsd
           time = getdata(file_list[idx*N + j])[8] - time0
        plt.loglog(freq, tot_psd/N, label = aux+aux_press+v, color = c)
        name = str(aux)+str(aux_press)+str(v)+" time= "+str(time)
        name = os.path.join(path, name)
        if savetxt:
            np.savetxt(name, (freq, tot_psd/N))
        #plt.loglog(freq, tot_psd/N, color = c)
        # plt.loglog(freq, tot_psd/N, label = aux+aux_press+aux_angle+aux2, color = c)
        plt.legend(loc="upper right", frameon = False)
plt.ylabel(r'Sensor response [$V^2/Hz$]', fontsize = 20)
plt.xlabel(r'Frequency [Hz]', fontsize = 20)
plt.rcParams.update({'font.size':20})
plt.show()
