from numpy import log, sqrt
import scipy.constants

ccm = scipy.constants.c * 100 #speed of light in cm/s
cm = scipy.constants.c #speed of light in m/s
ckm = scipy.constants.c * 0.001 #speed of light in km/s
h = scipy.constants.h #Planck's constant in Js
k = scipy.constants.k #Boltzmann's constant in J/K
kcm = scipy.constants.k * (scipy.constants.h**-1) * ((scipy.constants.c * 100)**-1) #Boltzmann's constants in cm-1/K
sigma_to_fwhm = 2.0*sqrt(2.0*log(2.0))
fwhm_to_sigma = 1.0/sigma_to_fwhm
