#required libraries
import urllib
import scipy
import pydub
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft as fft




#a temp folder for downloads
temp_folder="C:\PYTHON PROJECTS"

#spotify mp3 sample file
#web_file="http://p.scdn.co/mp3-preview/35b4ce45af06203992a86fa729d17b1c1f93cac5"

sound = pydub.AudioSegment.from_mp3("A.mp3")
sound.export("C:\PYTHON PROJECTS\A.wav", format="wav")

rate,audData=scipy.io.wavfile.read("A.wav")
scipy.io.wavfile.write("B.wav", 176400, audData)
#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate

Len = audData.shape[0] / rate
channel1=audData[:,0] #left
channel2=audData[:,1] #right
print(rate)
print(audData)
print ("DLUGOSC = ")
print (Len)
print(time)
fourier=fft.fft(channel1)

n = len(channel1)



fourier = fourier[0:(n//2)]

fourier = fourier / float(n)

freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);

plt.figure(3)
plt.plot(freqArray/1000, 10*np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')



plt.figure(1)
plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')

plt.figure(4, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
#plt.show()

#plot amplitude (or loudness) over time
plt.figure(2)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.01, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.subplot(212)
#plt.plot(time, channel2, linewidth=0.001, alpha=0.7, color='#ff7f00')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
plt.show()
