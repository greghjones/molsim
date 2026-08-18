#include "sleefinline_purecfma_scalar.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
    // Fallback for sequential execution
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
#endif

#include "util.hpp"
#include "constants.hpp"

namespace molsim::functional::detail
{
    void calc_tau_purecpp(const AlignedVector<double>& aij,
                        const AlignedVector<int>& gup,
                        const AlignedVector<double>& eup,
                        const AlignedVector<double>& frequencies,
                        const double columndensity,
                        const double Tex,
                        const double dV,
                        const double q,
                              AlignedVector<double>& tau)
    {
        const double texinv = -1.0/Tex;
        const double boltzmannscale = h*1.0e6/(k*Tex);
        const double prefactor = std::sqrt(std::log(2.0)) * cm*cm*cm * (columndensity * 100*100) 
                                / (4*std::pow(std::numbers::pi,1.5) * 1.0e18 * dV * 1000.0 * q);

        const auto len = frequencies.size();
        tau.resize(len);

        for (auto& element : tau) element = 0.0;

        for (long i = 0; i < len; i++)
        {
            double exp1 = Sleef_expd1_u10purecfma(eup[i]*texinv);
            double exp2 = Sleef_expm1d1_u10purecfma(frequencies[i]*boltzmannscale);
            double finv = prefactor/(frequencies[i]*frequencies[i]*frequencies[i]);
            tau[i] = (aij[i]*gup[i])*(exp1*exp2)*finv;
        }
    }

    void make_gaussians_purecpp(const AlignedVector<double>& centers,
                              const AlignedVector<double>& int0s,
                              const AlignedVector<ssize_t>& lls,
                              const AlignedVector<ssize_t>& uls, 
                              const double dV,
                              const AlignedVector<double>& x,
                                    AlignedVector<double>& y)
    {
        auto npeaks = centers.size();

        y.resize(x.size());
        for (auto& element : y) element = 0.0;

        assert(int0s.size() == npeaks);
        assert(lls.size() == npeaks);
        assert(uls.size() == npeaks);

        const double scale1 = 2*(dV/ckm*fwhm_to_sigma)*(dV/ckm*fwhm_to_sigma);

        for (long n = 0; n < npeaks; n++)
        {
            const double center = centers[n];
            const double int0 = int0s[n];
            const double scale2 = -1.0 / (scale1*center*center);

            for (long i = lls[n]; i < uls[n]; i++)
            {
                double v = x[i] - center;
                v = Sleef_expd1_u10purecfma(scale2*v*v);
                y[i] += int0*v;
            }
        }      
    }

    void calc_Tb_purecpp(const AlignedVector<double>& frequency,
                       const AlignedVector<double>& tau,
                       const AlignedVector<double>& Tbg,
                       const double Tex,
                             AlignedVector<double>& Tb)
    {
        const auto len = frequency.size();
        const double texinv = 1.0/Tex;
        const double scale = h*1.0e6/k;

        assert(Tbg.size() == len);
        assert(tau.size() == len);

        Tb.resize(len);

        #pragma omp parallel
        {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        const ssize_t offset = len / nthreads;
        const ssize_t localoffset = offset*tid;
        const ssize_t todo = (tid == nthreads-1) ? len-localoffset : offset;
        const double* __restrict pfreq = frequency.data()+localoffset;
        const double* __restrict ptau  = tau.data()+localoffset;
        const double* __restrict ptbg  = Tbg.data()+localoffset;
              double* __restrict ptb   = Tb.data()+localoffset;

        for (ssize_t i = 0; i < todo; i++)
        {
            double temp1 = (*pfreq)*scale;
            double j_t = Sleef_expm1d1_u10purecfma(temp1*texinv);
            double j_tbg = Sleef_expm1d1_u10purecfma(temp1/(*ptbg));
            double tau = Sleef_expm1d1_u10purecfma(-(*ptau));
            j_t = temp1/j_t;
            j_tbg = temp1/j_tbg;
            *ptb = tau*(j_tbg - j_t);
            pfreq++;
            ptbg++;
            ptau++;
            ptb++;
        }
        }
    }

    void calc_Tb_purecpp(const AlignedVector<double>& frequency,
                         const AlignedVector<double>& tau,
                         const double Tbg,
                         const double Tex,
                               AlignedVector<double>& Tb)
    {
        const auto len = frequency.size();
        const double texinv = 1.0/Tex;
        const double scale = h*1.0e6/k;
        const double tbginv = 1.0/Tbg;

        assert(tau.size() == len);

        Tb.resize(len);

        for (ssize_t i = 0; i < len; i++)
        {
            double temp1 = frequency[i]*scale;
            double j_t = std::expm1(temp1*texinv);
            double j_tbg = std::expm1(temp1*tbginv);
            double exptau = std::expm1(-tau[i]);
            j_t = temp1/j_t;
            j_tbg = temp1/j_tbg;
            Tb[i] = exptau*(j_tbg - j_t);
        }
    }

    std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_purecpp(const AlignedVector<double>& freq_array,
                                                                            const AlignedVector<double>& int_arr,
                                                                            const double source_size,
                                                                            const double dish_size)
    {
        const double beam_size_factor = 206265.0 * 1.22 * (cm/1.0e6) / dish_size;
        const double ss2 = source_size*source_size;

        const auto len = freq_array.size();

        assert(int_arr.size() == len);

        AlignedVector<double> result(len);
        AlignedVector<double> beam_dilution(len);

        for (long i = 0; i < len; i++)
        {
            const double beam_size = beam_size_factor / freq_array[i];
            beam_dilution[i] = ss2 / ((beam_size*beam_size) + ss2);
            result[i] = int_arr[i]*beam_dilution[i];
        }

        return {result, beam_dilution};
    }
}