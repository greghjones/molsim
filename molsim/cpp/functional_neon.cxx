#if defined(__ARM_NEON)

#include <arm_neon.h>
#include "sleefinline_advsimd.hpp"
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
    void calc_tau_neon(const AlignedVector<double>& aij,
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

        std::println("len: {}", len);

        for (auto& element : tau) element = 0.0;

        const int* __restrict pgup = gup.data();
        const double* __restrict paij = aij.data();
        const double* __restrict peup = eup.data();
        const double* __restrict pfreq = frequencies.data();
              double* __restrict ptau = tau.data();

        const long vecsize = 2;
        const long maxit = len/vecsize;
        const long remainder = len % vecsize;
        auto vtexinv = vdupq_n_f64(texinv);
        auto vboltzmann = vdupq_n_f64(boltzmannscale);
        auto vprefactor = vdupq_n_f64(prefactor);
        std::println("maxit: {}", maxit);
        for (long i = 0; i < maxit; i++)
        {
            auto exp1 = vld1q_f64(peup);
            exp1 = vmulq_f64(vtexinv, exp1);
            exp1 = Sleef_expd2_u10advsimd(exp1);
            auto f = vld1q_f64(pfreq);
            auto exp2 = vmulq_f64(vboltzmann, f);
            exp2 = Sleef_expm1d2_u10advsimd(exp2);
            auto finv = vmulq_f64(f, f);
            finv = vmulq_f64(f, finv);
            finv = vdivq_f64(vprefactor, finv);
            auto gupint = vld1_s32(pgup);
            auto gupf32 = vcvt_f32_s32(gupint);
            auto gup = vcvt_f64_f32(gupf32);
            auto res = vld1q_f64(paij);
            res = vmulq_f64(res,gup);
            exp1 = vmulq_f64(exp1, exp2);
            res = vmulq_f64(res,finv);
            res = vmulq_f64(res,exp1);
            vst1q_f64(ptau, res);
            pgup  += vecsize;
            peup  += vecsize;
            paij  += vecsize;
            pfreq += vecsize;
            ptau  += vecsize;
        }
        for (long i = 0; i < remainder; i++)
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

    void make_gaussians_neon(const AlignedVector<double>& centers,
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

            const long vecsize = 2;
            const long alignedll = (lls[n]/vecsize)*vecsize;
            const long len = uls[n] - alignedll;
            const long unrollfac = 4;
            const long itlen = vecsize*unrollfac;
            const long maxit = len/itlen;
            const long remainder = len % itlen;
            auto vs = vdupq_n_f64(scale2);
            auto vc = vdupq_n_f64(center);
            auto va = vdupq_n_f64(int0);

            const double* __restrict px = x.data() + alignedll;
                  double* __restrict py = y.data() + alignedll;

            for (long i = 0; i < maxit; i++)
            {
                auto vinc1 = vld1q_f64(px);
                auto vinc2 = vld1q_f64(px+vecsize);
                auto vinc3 = vld1q_f64(px+2*vecsize);
                auto vinc4 = vld1q_f64(px+3*vecsize);

                auto vy1   = vld1q_f64(py);
                auto vy2   = vld1q_f64(py+vecsize);
                auto vy3   = vld1q_f64(py+2*vecsize);
                auto vy4   = vld1q_f64(py+3*vecsize);

                vinc1 = vsubq_f64(vinc1, vc);
                vinc2 = vsubq_f64(vinc2, vc);
                vinc3 = vsubq_f64(vinc3, vc);
                vinc4 = vsubq_f64(vinc4, vc);

                vinc1 = vmulq_f64(vinc1, vinc1);
                vinc2 = vmulq_f64(vinc2, vinc2);
                vinc3 = vmulq_f64(vinc3, vinc3);
                vinc4 = vmulq_f64(vinc4, vinc4);

                vinc1 = vmulq_f64(vs, vinc1);
                vinc2 = vmulq_f64(vs, vinc2);
                vinc3 = vmulq_f64(vs, vinc3);
                vinc4 = vmulq_f64(vs, vinc4);

                vinc1 = Sleef_expd2_u10advsimd(vinc1);
                vinc2 = Sleef_expd2_u10advsimd(vinc2);
                vinc3 = Sleef_expd2_u10advsimd(vinc3);
                vinc4 = Sleef_expd2_u10advsimd(vinc4);

                vy1   = vfmaq_f64(vy1, va, vinc1);
                vy2   = vfmaq_f64(vy2, va, vinc2);
                vy3   = vfmaq_f64(vy3, va, vinc3);
                vy4   = vfmaq_f64(vy4, va, vinc4);

                vst1q_f64(py          , vy1);
                vst1q_f64(py+  vecsize, vy2);
                vst1q_f64(py+2*vecsize, vy3);
                vst1q_f64(py+3*vecsize, vy4);
                px += itlen;
                py += itlen;
            }
            for (long i = 0; i < remainder; i++)
            {
                double v = *px - center;
                v = Sleef_expd1_u10purecfma(scale2*v*v);
                *py += int0*v;
                px++;
                py++;
            }
        }
    }

    void calc_Tb_neon(const AlignedVector<double>& frequency,
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

        const long vecsize = 2;
        const long unrollfactor = 4;
        const long itlen = vecsize*unrollfactor;

        const long offset = ((len/itlen) / nthreads)*itlen;
        const long localoffset = offset*tid;
        const double* __restrict pfreq = frequency.data()+localoffset;
        const double* __restrict ptau = tau.data()+localoffset;
        const double* __restrict ptbg = Tbg.data()+localoffset;
            double* __restrict ptb = Tb.data()+localoffset;

        const long ntodo = (tid == nthreads-1) ? len-localoffset : offset;
        const long maxit = ntodo/itlen;
        const long remainder = ntodo % itlen;
        const auto vtexinv = vdupq_n_f64(texinv);
        const auto vscale = vdupq_n_f64(scale);
        const auto vm1 = vdupq_n_f64(-1.0);
        for (long i = 0; i < maxit; i++)
        {
            auto freq = vld1q_f64(pfreq);
            auto tbg = vld1q_f64(ptbg);
            auto tau = vld1q_f64(ptau);
            auto temp1 = vmulq_f64(freq, vscale);
            tau = vmulq_f64(tau, vm1);
            auto j_t = vmulq_f64(temp1, vtexinv);
            auto j_tbg = vdivq_f64(temp1, tbg);
            tau = Sleef_expm1d2_u10advsimd(tau);
            j_t = Sleef_expm1d2_u10advsimd(j_t);
            j_tbg = Sleef_expm1d2_u10advsimd(j_tbg);
            j_t = vdivq_f64(temp1, j_t);
            j_tbg = vdivq_f64(temp1, j_tbg);
            auto res = vsubq_f64(j_tbg, j_t);
            res = vmulq_f64(tau, res);
            vst1q_f64(ptb, res);
            pfreq += itlen;
            ptbg  += itlen;
            ptau  += itlen;
            ptb   += itlen;
        }
        for (long i = 0; i < remainder; i++)
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

    void calc_Tb_neon(const AlignedVector<double>& frequency,
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

        const long vecsize = 2;
        const long unrollfactor = 4;
        const long itlen = vecsize*unrollfactor;

        const long offset = ((len/itlen) / nthreads)*itlen;
        const long localoffset = offset*tid;
        const double* __restrict pfreq = frequency.data()+localoffset;
        const double* __restrict ptau = tau.data()+localoffset;
              double* __restrict ptb = Tb.data()+localoffset;

        const long ntodo = (tid == nthreads-1) ? len-localoffset : offset;
        const long maxit = ntodo/itlen;
        const long remainder = ntodo % itlen;
        const auto vtexinv = vdupq_n_f64(texinv);
        const auto vscale = vdupq_n_f64(scale);
        const auto tbg = vdupq_n_f64(Tbg);
        for (long i = 0; i < maxit; i++)
        {
            auto freq1 = vld1q_f64(pfreq);
            auto freq2 = vld1q_f64(pfreq+  vecsize);
            auto freq3 = vld1q_f64(pfreq+2*vecsize);
            auto freq4 = vld1q_f64(pfreq+3*vecsize);

            auto tau1 = vld1q_f64(ptau);
            auto tau2 = vld1q_f64(ptau+  vecsize);
            auto tau3 = vld1q_f64(ptau+2*vecsize);
            auto tau4 = vld1q_f64(ptau+3*vecsize);

            auto temp1 = vmulq_f64(freq1, vscale);
            auto temp2 = vmulq_f64(freq2, vscale);
            auto temp3 = vmulq_f64(freq3, vscale);
            auto temp4 = vmulq_f64(freq4, vscale);

            tau1 = vnegq_f64(tau1);
            tau2 = vnegq_f64(tau2);
            tau3 = vnegq_f64(tau3);
            tau4 = vnegq_f64(tau4);

            auto j_t1 = vmulq_f64(temp1, vtexinv);
            auto j_t2 = vmulq_f64(temp2, vtexinv);
            auto j_t3 = vmulq_f64(temp3, vtexinv);
            auto j_t4 = vmulq_f64(temp4, vtexinv);

            auto j_tbg1 = vdivq_f64(temp1, tbg);
            auto j_tbg2 = vdivq_f64(temp2, tbg);
            auto j_tbg3 = vdivq_f64(temp3, tbg);
            auto j_tbg4 = vdivq_f64(temp4, tbg);

            tau1 = Sleef_expm1d2_u10advsimd(tau1);
            tau2 = Sleef_expm1d2_u10advsimd(tau2);
            tau3 = Sleef_expm1d2_u10advsimd(tau3);
            tau4 = Sleef_expm1d2_u10advsimd(tau4);
            
            j_t1 = Sleef_expm1d2_u10advsimd(j_t1);
            j_t2 = Sleef_expm1d2_u10advsimd(j_t2);
            j_t3 = Sleef_expm1d2_u10advsimd(j_t3);
            j_t4 = Sleef_expm1d2_u10advsimd(j_t4);

            j_tbg1 = Sleef_expm1d2_u10advsimd(j_tbg1);
            j_tbg2 = Sleef_expm1d2_u10advsimd(j_tbg2);
            j_tbg3 = Sleef_expm1d2_u10advsimd(j_tbg3);
            j_tbg4 = Sleef_expm1d2_u10advsimd(j_tbg4);

            j_t1 = vdivq_f64(temp1, j_t1);
            j_t2 = vdivq_f64(temp2, j_t2);
            j_t3 = vdivq_f64(temp3, j_t3);
            j_t4 = vdivq_f64(temp4, j_t4);

            j_tbg1 = vdivq_f64(temp1, j_tbg1);
            j_tbg2 = vdivq_f64(temp2, j_tbg2);
            j_tbg3 = vdivq_f64(temp3, j_tbg3);
            j_tbg4 = vdivq_f64(temp4, j_tbg4);

            auto res1 = vsubq_f64(j_tbg1, j_t1);
            auto res2 = vsubq_f64(j_tbg2, j_t2);
            auto res3 = vsubq_f64(j_tbg3, j_t3);
            auto res4 = vsubq_f64(j_tbg4, j_t4);

            res1 = vmulq_f64(tau1, res1);
            res2 = vmulq_f64(tau2, res2);
            res3 = vmulq_f64(tau3, res3);
            res4 = vmulq_f64(tau4, res4);

            vst1q_f64(ptb          , res1);
            vst1q_f64(ptb+  vecsize, res2);
            vst1q_f64(ptb+2*vecsize, res3);
            vst1q_f64(ptb+3*vecsize, res4);

            pfreq += itlen;
            ptau += itlen;
            ptb += itlen;
        }

        for (long i = 0; i < remainder; i++)
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

    std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_neon(const AlignedVector<double>& freq_array,
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

        const long vecsize = 2;
        const long unrollfac = 2;
        const long itlen = vecsize*unrollfac;
        const long maxit = len/itlen;
        const long remainder = len % itlen;
        auto fac = vdupq_n_f64(beam_size_factor);
        auto vss2 = vdupq_n_f64(ss2);
        for (long i = 0; i < maxit; i++)
        {
            auto f1 = vld1q_f64(pf);
            auto f2 = vld1q_f64(pf+vecsize);

            auto vi1 = vld1q_f64(pint);
            auto vi2 = vld1q_f64(pint+vecsize);

            auto bs1 = vdivq_f64(fac, f1);
            auto bs2 = vdivq_f64(fac, f2);

            auto bd1 = vfmaq_f64(vss2, bs1, bs1);
            auto bd2 = vfmaq_f64(vss2, bs2, bs2);

                bd1 = vdivq_f64(vss2, bd1);
                bd2 = vdivq_f64(vss2, bd2);

            auto res1 = vmulq_f64(vi1,bd1);
            auto res2 = vmulq_f64(vi2,bd2);

            vst1q_f64(pbd        , bd1);
            vst1q_f64(pbd+vecsize, bd2);

            vst1q_f64(pr        , res1);
            vst1q_f64(pr+vecsize, res2);

            pf += itlen;
            pint += itlen;
            pr += itlen;
            pbd += itlen;
        }
        for (long i = 0; i < remainder; i++)
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
