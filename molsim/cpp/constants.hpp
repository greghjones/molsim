#ifndef _MOLSIM_CONSTANTS_H
#define _MOLSIM_CONSTANTS_H

#include <numbers>
#include <cmath>

const double h = 6.62607015e-34;
const double k = 1.380649e-23;
const double cm = 299792458;
const double ckm = cm*0.001;
const double sigma_to_fwhm = 2.0*std::sqrt(2.0*std::log(2.0));
const double fwhm_to_sigma = 1.0/sigma_to_fwhm;

#endif
