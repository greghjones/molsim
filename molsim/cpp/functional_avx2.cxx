#if defined(__AVX2__)

#include <emmintrin.h>
#include <immintrin.h>
#include "sleefinline_avx2.h"
#include "sleefinline_purecfma_scalar.hpp"

#if defined(_OPENMP)
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
    void calc_tau_avx2(const AlignedVector<double>& aij,
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

        const int* __restrict pgup = gup.data();
        const double* __restrict paij = aij.data();
        const double* __restrict peup = eup.data();
        const double* __restrict pfreq = frequencies.data();
              double* __restrict ptau = tau.data();


        const ssize_t vecsize = 4;
        const ssize_t unrollfactor = 1;
        const ssize_t itlen = vecsize*unrollfactor;
        const ssize_t maxit = len/itlen;
        const ssize_t remainder = len % itlen;
        auto pgup2 = reinterpret_cast<const __m128i* __restrict>(pgup);
        auto vtexinv    = _mm256_set1_pd(texinv);
        auto vboltzmann = _mm256_set1_pd(boltzmannscale);
        auto vprefactor = _mm256_set1_pd(prefactor);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto exp1 = _mm256_load_pd(peup);
            exp1 = _mm256_mul_pd(vtexinv, exp1);
            exp1 = Sleef_expd4_u10avx2(exp1);
            auto f = _mm256_load_pd(pfreq);
            auto exp2 = _mm256_mul_pd(vboltzmann, f);
            exp2 = Sleef_expm1d4_u10avx2(exp2);
            auto finv = _mm256_mul_pd(f, f);
            finv = _mm256_mul_pd(finv, f);
            finv = _mm256_div_pd(vprefactor, finv);
            auto gupint = _mm_load_si128(pgup2);
            auto gup = _mm256_cvtepi32_pd(gupint);
            auto res = _mm256_load_pd(paij);
            res = _mm256_mul_pd(res, gup);
            exp1 = _mm256_mul_pd(exp1, exp2);
            res = _mm256_mul_pd(res, finv);
            res = _mm256_mul_pd(res, exp1);
            _mm256_store_pd(ptau, res);
            pgup2  += unrollfactor; // due to pointer type
            peup  += itlen;
            paij  += itlen;
            pfreq += itlen;
            ptau  += itlen;
        }

        pgup = reinterpret_cast<const int* __restrict>(pgup2);

        for (ssize_t i = 0; i < remainder; i++)
        {
            double exp1 = Sleef_expd1_u10purecfma((*peup)*texinv);
            double f = (*pfreq);
            double exp2 = Sleef_expm1d1_u10purecfma(f*boltzmannscale);
            double finv = prefactor/(f*f*f);
            *ptau = ((*paij)*(*pgup))*(exp1*exp2)*finv;
            pgup++;
            peup++;
            paij++;
            pfreq++;
            ptau++;
        }
    }

    void make_gaussians_avx2(const AlignedVector<double>& centers,
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

        for (ssize_t n = 0; n < npeaks; n++)
        {
            const double center = centers[n];
            const double int0 = int0s[n];
            const double scale2 = -1.0 / (scale1*center*center);

            const ssize_t vecsize = 4;
            const ssize_t alignedll = (lls[n]/vecsize)*vecsize;
            const auto len = uls[n] - alignedll;
            const ssize_t maxit = len/vecsize;
            const ssize_t remainder = len % vecsize;

            const double* __restrict px = x.data() + alignedll;
                  double* __restrict py = y.data() + alignedll;

            auto vs = _mm256_set1_pd(scale2);
            auto vc = _mm256_set1_pd(center);
            auto va = _mm256_set1_pd(int0);
            for (ssize_t i = 0; i < maxit; i++)
            {
                auto vinc = _mm256_load_pd(px);
                auto vy   = _mm256_load_pd(py);
                     vinc = _mm256_sub_pd(vinc, vc);
                     vinc = _mm256_mul_pd(vinc, vinc);
                     vinc = _mm256_mul_pd(vs, vinc);
                     vinc = Sleef_expd4_u10avx2(vinc);
                     vy   = _mm256_fmadd_pd(va, vinc, vy);
                     _mm256_store_pd(py, vy);
                     px += vecsize;
                     py += vecsize;
            }

            for (ssize_t i = 0; i < remainder; i++)
            {
                double v = *px - center;
                v = Sleef_expd1_u10purecfma(scale2*v*v);
                *py += int0*v;
                px++;
                py++;
            }
        }   
    }

    void calc_Ibg_avx2(const AlignedVector<double>& freq,
                       const double Tbg,
                             AlignedVector<double>& Ibg)
    {
        const auto len = freq.size();
        if (Ibg.size() != len) Ibg.resize(len);

        const double f = 2.0e26 / (cm * cm);

        const double* __restrict pfreq = freq.data();
              double* __restrict pibg  = Ibg.data();

        const ssize_t vecsize = 4;
        const ssize_t unrollfactor = 1;
        const ssize_t itlen = vecsize*unrollfactor;
        const auto maxit = len/itlen;
        const auto remainder = len % itlen;
        const auto vf = _mm256_set1_pd(f);
        const auto vmhz = _mm256_set1_pd(1.0e6);
        const auto vh = _mm256_set1_pd(h);
        const double s3 = 1.0/(k*Tbg);
        const auto v3 = _mm256_set1_pd(s3);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto v0 = _mm256_load_pd(pfreq);
            auto v1 = _mm256_mul_pd(v0, vmhz);
            auto v2 = _mm256_mul_pd(v1, vh);
            auto v4 = _mm256_mul_pd(v2, v3);
            auto v5 = _mm256_mul_pd(v1, v2);
                 v5 = _mm256_mul_pd(v1, v5);
            auto v6 = Sleef_expm1d4_u10avx2(v4);
            auto v7 = _mm256_mul_pd(vf, v5);
            auto res = _mm256_div_pd(v7, v6);
            _mm256_store_pd(pibg, res);
            pfreq += itlen;
            pibg  += itlen;
        }
        for (ssize_t i = 0; i < remainder; i++)
        {
            double v1 = (*pfreq)*1.0e6;
            double v2 = v1*h;
            double v4 = v2*s3;
            double v5 = v1*v1*v2;
            double v6 = std::expm1(v4);
            double v7 = f*v5;
            *pibg = v7/v6;
            pfreq++;
            pibg++;
        }
    }

    void calc_Ibg_avx2(const AlignedVector<double>& freq,
                       const AlignedVector<double>& Tbg,
                             AlignedVector<double>& Ibg)
    {
        const auto len = freq.size();
        if (Ibg.size() != len) Ibg.resize(len);

        const double f = 2.0e26 / (cm * cm);

        const double* __restrict pfreq = freq.data();
        const double* __restrict ptbg  = Tbg.data();
              double* __restrict pibg  = Ibg.data();

        const ssize_t vecsize = 4;
        const ssize_t unrollfactor = 1;
        const ssize_t itlen = vecsize*unrollfactor;
        const auto maxit = len/itlen;
        const auto remainder = len % itlen;
        const auto vf = _mm256_set1_pd(f);
        const auto vmhz = _mm256_set1_pd(1.0e6);
        const auto vh = _mm256_set1_pd(h);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto v0 = _mm256_load_pd(pfreq);
            auto vt = _mm256_load_pd(ptbg);
            auto v1 = _mm256_mul_pd(v0, vmhz);
            auto v2 = _mm256_mul_pd(v1, vh);
            auto v3 = _mm256_mul_pd(vt, vk);
            auto v4 = _mm256_div_pd(v2, v3);
            auto v5 = _mm256_mul_pd(v1, v2);
                v5 = _mm256_mul_pd(v1, v5);
            auto v6 = Sleef_expm1d4_u10avx2(v4);
            auto v7 = _mm256_mul_pd(vf, v5);
            auto res = _mm256_div_pd(v7, v6);
            _mm256_store_pd(pibg, res);
            pfreq += vecsize;
            ptbg  += vecsize;
            pibg  += vecsize;
        }
        for (ssize_t i = 0; i < remainder; i++)
        {
            double v1 = (*pfreq)*1.0e6;
            double v2 = v1*h;
            double v3 = (*ptbg)*k;
            double v4 = v2/v3;
            double v5 = v1*v1*v2;
            double v6 = std::expm1(v4);
            double v7 = f*v5;
            *pibg = v7/v6;
            pfreq++;
            ptbg++;
            pibg++;
        }
    }

    void calc_Tb_avx2(const AlignedVector<double>& frequency,
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

        const ssize_t vecsize = 4;
        const ssize_t unrollfactor = 1;
        const auto itlen = vecsize*unrollfactor;

        const ssize_t offset = ((len/itlen) / nthreads)*itlen;
        const ssize_t localoffset = offset*tid;

        const double* __restrict pfreq = frequency.data()+localoffset;
        const double* __restrict ptau = tau.data()+localoffset;
        const double* __restrict ptbg = Tbg.data()+localoffset;
              double* __restrict ptb = Tb.data()+localoffset;

        const ssize_t ntodo = (tid == nthreads-1) ? len-localoffset : offset;
        const ssize_t maxit = ntodo/itlen;
        const ssize_t remainder = ntodo % itlen;

        const auto vtexinv = _mm256_set1_pd(texinv);
        const auto vscale = _mm256_set1_pd(scale);
        const auto vm1 = _mm256_set1_pd(-1.0);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto freq = _mm256_load_pd(pfreq);
            auto tbg  = _mm256_load_pd(ptbg);
            auto tau  = _mm256_load_pd(ptau);
            auto temp1 = _mm256_mul_pd(freq, vscale);
            tau = _mm256_mul_pd(tau, vm1);
            auto j_t = _mm256_mul_pd(temp1, vtexinv);
            auto j_tbg = _mm256_div_pd(temp1, tbg);
            tau = Sleef_expm1d4_u10avx2(tau);
            j_t = Sleef_expm1d4_u10avx2(j_t);
            j_tbg = Sleef_expm1d4_u10avx2(j_tbg);
            j_t = _mm256_div_pd(temp1, j_t);
            j_tbg = _mm256_div_pd(temp1, j_tbg);
            auto res = _mm256_sub_pd(j_tbg, j_t);
            res = _mm256_mul_pd(res, tau);
            _mm256_store_pd(ptb, res);
            pfreq += itlen;
            ptbg  += itlen;
            ptau  += itlen;
            ptb   += itlen;
        }
        
        for (ssize_t i = 0; i < remainder; i++)
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

    void calc_Tb_avx2(const AlignedVector<double>& frequency,
                      const AlignedVector<double>& tau,
                      const double Tbg,
                      const double Tex,
                            AlignedVector<double>& Tb)
    {
        const auto len = frequency.size();
        const double texinv = 1.0/Tex;
        const double scale = h*1.0e6/k;

        assert(tau.size() == len);

        Tb.resize(len);

        #pragma omp parallel
        {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        const ssize_t vecsize = 4;
        const ssize_t unrollfactor = 1;
        const auto itlen = vecsize*unrollfactor;

        const ssize_t offset = ((len/itlen) / nthreads)*itlen;
        const ssize_t localoffset = offset*tid;

        const double* __restrict pfreq = frequency.data()+localoffset;
        const double* __restrict ptau = tau.data()+localoffset;
            double* __restrict ptb = Tb.data()+localoffset;

        const ssize_t ntodo = (tid == nthreads-1) ? len-localoffset : offset;
        const ssize_t maxit = ntodo/itlen;
        const ssize_t remainder = ntodo % itlen;

        const auto vtexinv = _mm256_set1_pd(texinv);
        const auto vscale = _mm256_set1_pd(scale);
        const auto vm1 = _mm256_set1_pd(-1.0);
        const auto tbg = _mm256_set1_pd(Tbg);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto freq = _mm256_load_pd(pfreq);
            auto tau  = _mm256_load_pd(ptau);
            auto temp1 = _mm256_mul_pd(freq, vscale);
            tau = _mm256_mul_pd(tau, vm1);
            auto j_t = _mm256_mul_pd(temp1, vtexinv);
            auto j_tbg = _mm256_div_pd(temp1, tbg);
            tau = Sleef_expm1d4_u10avx2(tau);
            j_t = Sleef_expm1d4_u10avx2(j_t);
            j_tbg = Sleef_expm1d4_u10avx2(j_tbg);
            j_t = _mm256_div_pd(temp1, j_t);
            j_tbg = _mm256_div_pd(temp1, j_tbg);
            auto res = _mm256_sub_pd(j_tbg, j_t);
            res = _mm256_mul_pd(res, tau);
            _mm256_store_pd(ptb, res);
            pfreq += itlen;
            ptau  += itlen;
            ptb   += itlen;
        }
        for (ssize_t i = 0; i < remainder; i++)
        {
            double temp1 = (*pfreq)*scale;
            double j_t = Sleef_expm1d1_u10purecfma(temp1*texinv);
            double j_tbg = Sleef_expm1d1_u10purecfma(temp1/Tbg);
            double tau = Sleef_expm1d1_u10purecfma(-(*ptau));
            j_t = temp1/j_t;
            j_tbg = temp1/j_tbg;
            *ptb = tau*(j_tbg - j_t);
            pfreq++;
            ptau++;
            ptb++;
        }
        }
    }

    std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_avx2(const AlignedVector<double>& freq_array,
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

        const double* __restrict pf = freq_array.data();
        const double* __restrict pint = int_arr.data();
              double* __restrict pr = result.data();
              double* __restrict pbd = beam_dilution.data();

        const ssize_t vecsize = 4;
        const ssize_t unrollfac = 2;
        const ssize_t itlen = vecsize*unrollfac;
        const ssize_t maxit = len/itlen;
        const ssize_t remainder = len % itlen;
        auto fac = _mm256_set1_pd(beam_size_factor);
        auto vss2 = _mm256_set1_pd(ss2);
        for (ssize_t i = 0; i < maxit; i++)
        {
            auto f1 = _mm256_load_pd(pf);
            auto f2 = _mm256_load_pd(pf+vecsize);

            auto vi1 = _mm256_load_pd(pint);
            auto vi2 = _mm256_load_pd(pint+vecsize);

            auto bs1 = _mm256_div_pd(fac, f1);
            auto bs2 = _mm256_div_pd(fac, f2);

            auto bd1 = _mm256_fmadd_pd(bs1, bs1, vss2);
            auto bd2 = _mm256_fmadd_pd(bs2, bs2, vss2);

                bd1 = _mm256_div_pd(vss2, bd1);
                bd2 = _mm256_div_pd(vss2, bd2);

            auto res1 = _mm256_mul_pd(vi1, bd1);
            auto res2 = _mm256_mul_pd(vi2, bd2);

            _mm256_store_pd(pbd        , bd1);
            _mm256_store_pd(pbd+vecsize, bd2);

            _mm256_store_pd(pr        , res1);
            _mm256_store_pd(pr+vecsize, res2);

            pf += itlen;
            pint += itlen;
            pr += itlen;
            pbd += itlen;
        }
        for (ssize_t i = 0; i < remainder; i++)
        {
            const double beam_size = beam_size_factor / (*pf);
            const double beam_dilution = ss2 / ((beam_size*beam_size) + ss2);
            *pbd = beam_dilution;
            *pr = (*pint)*beam_dilution;
            pf++;
            pint++;
            pr++;
            pbd++;
        }

        return {result, beam_dilution};
    }
}



#endif
