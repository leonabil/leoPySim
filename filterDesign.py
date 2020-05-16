# The following is a Python/scipy snippet to generate the
# coefficients for a halfband filter.  A halfband filter
# is a filter where the cutoff frequency is Fs/4 and every
# other coeffecient is zero except the cetner tap.
# Note: every other (even except 0) is 0, most of the coefficients
#       will be close to zero, force to zero actual

import numpy
from numpy import log10, abs, pi
import scipy
from scipy import signal
import matplotlib
import matplotlib.pyplot as mpl

# ~~[Filter Design with Windowed freq]~~
N1=11
flatBand = 0.7e6
b = signal.firwin(N1+1, flatBand, fs=8e6)
(wb, Hb) = signal.freqz(b)
print ('N = %1d', N1)
for ii in range(N1+1):
    print(' tap %2d   %-3.6f' % (ii, b[ii]))

N2=13
b2= signal.firwin(N2+1, flatBand, fs=8e6)
(wb2, Hb2) = signal.freqz(b2)
print ('N = %1d', N2)
for ii in range(N2+1):
print(' tap %2d   %-3.6f' % (ii, b2[ii]))
#b[abs(h) <= 1e-4] = 0.


# Dump the coefficients for comparison and verification


## ~~[Plotting]~~
# Note: the pylab functions can be used to create plots,
#       and these might be easier for beginners or more familiar
#       for Matlab users.  pylab is a wrapper around lower-level
#       MPL artist (pyplot) functions.
#fig = mpl.pyplot.figure()
#ax0 = fig.add_subplot(211)
#ax0.stem(numpy.arange(len(h)), h)
#ax0.grid(True)
#ax0.set_title('Parks-McClellan (remez) Impulse Response')
#ax1 = fig.add_subplot(212)
#ax1.stem(numpy.arange(len(b)), b)
#ax1.set_title('Windowed Frequency Sampling (firwin) Impulse Response')
#ax1.grid(True)
#fig.savefig('hb_imp.png')

fig = mpl.figure()
ax1 = fig.add_subplot(121)
ax1.plot(wb*8/(2*pi), 20*log10(abs(Hb)))

#bx = bands*2*pi
#ax1.axvspan(bx[1], bx[2], facecolor='0.5', alpha='0.33')
#ax1.plot(pi/2, -6, 'go')
#ax1.axvline(pi/2, color='g', linestyle='--')
#ax1.axis([0,pi,-64,3])
#ax1.grid('on')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_title(N1)
ax1.grid(True)
ax2 = fig.add_subplot(122)
ax2.plot(wb2*4/(2*pi), 20*log10(abs(Hb2)))
ax2.set_ylabel('Magnitude (dB)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title(N2)
ax2.grid(True)
#fig.savefig('hb_rsp.png')
mpl.show()