#include <Python.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <numbers>

#include <print>

#include "Sleef.h"

#include "util.hpp"
#include "constants.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<double> calc_tau_impl(AlignedVector<double> aij, AlignedVector<int> gup, AlignedVector<double> eup, AlignedVector<double> frequencies, double columndensity, double tex, double dV, double q)
{
    const double texinv = -1.0/tex;
    const double boltzmannscale = h*1.0e6/(k*tex);
    const double prefactor = std::sqrt(std::log(2.0)) * cm*cm*cm * (columndensity * 100*100) 
                             / (4*std::pow(std::numbers::pi,1.5) * 1.0e18 * dV * 1000.0 * q);

    const auto len = frequencies.size();

    AlignedVector<double> tau(aij.size(), 0.0);

    const    int* __restrict pgup = gup.data();
    const double* __restrict paij = aij.data();
    const double* __restrict peup = eup.data();
    const double* __restrict pfreq = frequencies.data();
          double* __restrict ptau = tau.data();


    #if defined(__AVX2__)
    const long vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
    auto vtexinv    = _mm256_set1_pd(texinv);
    auto vboltzmann = _mm256_set1_pd(boltzmannscale);
    auto vprefactor = _mm256_set1_pd(prefactor);
    for (long i = 0; i < maxit; i++)
    {
        auto exp1 = _mm256_set_pd(peup[3], peup[2], peup[1], peup[0]);
        exp1 = _mm256_mul_pd(vtexinv, exp1);
        exp1 = Sleef_expd4_u10avx2(exp1);
        auto f = _mm256_set_pd(pfreq[3], pfreq[2], pfreq[1], pfreq[0]);
        auto exp2 = _mm256_mul_pd(vboltzmann, f);
        exp2 = Sleef_expd4_u10avx2(exp2);
        auto finv = _mm256_mul_pd(f, f);
        finv = _mm256_mul_pd(finv, f);
        finv = _mm256_div_pd(vprefactor, finv);
        auto gupint = _mm_set_epi32(pgup[3], pgup[2], pgup[1], pgup[0]);
        auto gup = _mm256_cvtepi32_pd(gupint);
        auto res = _mm256_set_pd(paij[3], paij[2], paij[1], paij[0]);
        res = _mm256_mul_pd(res, gup);
        exp1 = _mm256_mul_pd(exp1, exp2);
        res = _mm256_mul_pd(res, finv);
        res = _mm256_mul_pd(res, exp1);
        _mm256_storeu_pd(ptau, res);
        pgup  += vecsize;
        paij  += vecsize;
        peup  += vecsize;
        pfreq += vecsize;
        ptau  += vecsize;
    }
    #elif defined(__ARM_NEON)
    const int vecsize = 2;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
    auto vtexinv = vdupq_n_f64(texinv);
    auto vboltzmann = vdupq_n_f64(boltzmannscale);
    auto vprefactor = vdupq_n_f64(prefactor);
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
        paij  += vecsize;
        peup  += vecsize;
        pfreq += vecsize;
        ptau  += vecsize;
    }
    #else
    const auto remainder = len;
    #endif
    for (long i = 0; i < remainder; i++)
    {
        double exp1 = Sleef_expd1_u10((*peup)*texinv);
        double f = (*pfreq);
        double exp2 = Sleef_expm1d1_u10(f*boltzmannscale);
        double finv = prefactor/(f*f*f);
        *ptau = ((*paij)*(*pgup))*(exp1*exp2)*finv;
        pgup++;
        paij++;
        peup++;
        paij++;
        pfreq++;
        ptau++;
    }

    return py::array_t {tau.size(), tau.data()};
}

py::array_t<double> calc_tau(const py::array_t<double>& aij, const py::array_t<int>& gup, const py::array_t<double>& eup, const py::array_t<double>& frequencies, double columndensity, double tex, double dV, double q)
{
    return calc_tau_impl(aij, gup, eup, frequencies, columndensity, tex, dV, q);
}

PYBIND11_MODULE(cxx, m, py::mod_gil_not_used())
{
    m.doc() = "Test C++ module";
    m.def("calc_tau", &calc_tau, "Test description", 
    "aij"_a, "gup"_a, "eup"_a, "frequencies"_a, "columndensity"_a, "tex"_a, "dV"_a, "q"_a);
}
