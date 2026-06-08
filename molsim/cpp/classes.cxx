#include <cstddef>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>
#include <print>

#include "sleef.h"

#include "classes.hpp"
#include "constants.hpp"
#include "util.hpp"

template <typename T, typename U, typename V>

static void __attribute__((always_inline)) vmulv(const AlignedVector<T>& a, const AlignedVector<U>& b, AlignedVector<V>& c, bool increment)
{
    assert(a.size() == b.size());
    assert(b.size() == c.size());

    if(increment)
    {
        #pragma omp simd
        for (long i = 0; i < a.size(); i++)
            c[i] += a[i]*b[i];
    }
    else
    {
        #pragma omp simd
        for (long i = 0; i < a.size(); i++)
            c[i] = a[i]*b[i];
    }
}

AlignedVector<double> _apply_vlsr(const AlignedVector<double>& infreq, double vlsr)
{
    const double scale = 1.0 - vlsr/ckm;

    AlignedVector<double> outfreq(infreq.size());

    // if we decide to have a BLAS dependency, replace with dscalv
    #pragma omp simd
    for (long i = 0; i < infreq.size(); i++)
        outfreq[i] = scale*infreq[i];

    return outfreq;
}

template <typename T1, typename T2>
inline std::vector<char> _trim_arr_mask(const AlignedVector<T1>& arr, const AlignedVector<T2>& lls, const AlignedVector<T2>& uls, const AlignedVector<T2>& key_arr)
{
    assert(lls.size() == uls.size());
    assert(arr.size() == key_arr.size());

    auto s = key_arr.size();

    std::vector<char> mask(arr.size(), false);

    for (long i = 0; i < lls.size(); i++)
    {
        const auto ll = lls[i];
        const auto ul = uls[i];
        assert(ll < ul);
        for (long j = 0; j < key_arr.size(); j++)
        {
            if ((key_arr[j] > ll) && (key_arr[j] < ul))
                mask[j] = true;
        }
    }

    return mask;
}

template <typename T>
AlignedVector<T> _apply_mask(const AlignedVector<T>& in, const std::vector<char>& mask)
{
    assert(mask.size() == in.size());
    AlignedVector<T> out {};

    for (long i = 0; i < mask.size(); i++)
        if (mask[i]) out.push_back(in[i]);

    return out;
}

Simulation::Simulation(Spectrum spectrum_,
                       std::optional<Observation> observation_,
                       Source source_,
                       AlignedVector<double> ll_,
                       AlignedVector<double> ul_,
                       const std::string& line_profile_,
                       double sim_width_,
                       double res_,
                       Molecule mol_,
                       const std::string& units_,
                       const std::string& notes_,
                       bool use_obs_,
                       bool add_noise_,
                       double noise_,
                       double tau_threshold_,
                       double eup_threshold_)
        : spectrum(spectrum_), observation(observation_), source(source_), ll(ll_), ul(ul_),
          sim_width(sim_width_), res(res_), mol(mol_), notes(notes_), use_obs(use_obs_), add_noise(add_noise_),
          noise(noise_), tau_threshold(tau_threshold_), eup_threshold(eup_threshold_)
        {
            set_line_profile(line_profile_);
            set_units(units_);
            set_arrays();
            if (spectrum.frequency.empty())
            {
                spectrum.freq_profile.clear();
                spectrum.int_profile.clear(); 
            }
        };

void Simulation::set_line_profile(std::string label)
{
    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c){ return std::tolower(c); });

    if (label == "gaussian") line_profile = Gaussian;
    else
    {
        std::print("Invalid line_profile type in Simulation!\n");
        // error macro
    }
};

void Simulation::set_units(std::string label)
{
    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c){ return std::tolower(c); });
    if (label == "k")
        units = K;
    else if (label == "mk")
        units = mK;
    else if (label == "jy/beam")
        units = Jy_beam;
    else
    {
        std::println("Invalid unit type for Simulation!");
        // error macro
    }
};

void Simulation::set_arrays()
{
    const double f = 1 - source.velocity/ckm;
    AlignedVector<double> tmp(mol.catalog.frequency.size());
    for (long i = 0; i < mol.catalog.frequency.size(); i++)
        tmp[i] = f*mol.catalog.frequency[i];

    auto mask = _trim_arr_mask(tmp, ll, ul, tmp);
    spectrum.frequency = _apply_mask(mol.catalog.frequency, mask);
    spectrum.freq0 = spectrum.frequency;
    aij = _apply_mask(mol.catalog.aij, mask);
    gup = _apply_mask(mol.catalog.gup, mask);
    eup = _apply_mask(mol.catalog.eup, mask);
}

void Simulation::apply_voffset()
{
    spectrum.frequency = _apply_vlsr(spectrum.freq0, source.velocity);
}

void Simulation::calc_tau()
{
    const double texinv = -1.0/source.Tex;
    const double boltzmannscale = h*1.0e6/(k*source.Tex);
    const double prefactor = std::sqrt(std::log(2.0)) * cm*cm*cm * (source.column * 100*100*100) 
                             / (4*std::pow(std::numbers::pi,1.5) * 1.0e18 * source.dV * 1000.0 * mol.q(source.Tex));

    const auto len = spectrum.frequency.size();
    spectrum.tau.resize(len);

    const    int* __restrict pgup = gup.data();
    const double* __restrict paij = aij.data();
    const double* __restrict peup = eup.data();
    const double* __restrict pfreq = spectrum.frequency.data();
          double* __restrict ptau = spectrum.tau.data();


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
        paij  += vecsize;
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
        paij  += vecsize;
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
}

void Simulation::calc_Tb()
{
    const auto len = spectrum.frequency.size();
    const double texinv = -1.0/source.Tex;
    const double scale = h*1.0e6/(k*source.Tex);

    const double* __restrict pfreq = spectrum.frequency.data();
    const double* __restrict ptbg = spectrum.Tbg_profile.data();
    const double* __restrict ptau = spectrum.tau_profile.data();
          double* __restrict ptb = spectrum.Tb.data();

    #if defined(__AVX2__)
    const long vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
    const auto vtexinv = _mm256_set1_pd(texinv);
    const auto vscale = _mm256_set1_pd(scale);
    const auto vm1 = _mm256_set1_pd(-1.0);
    for (long i = 0; i < maxit; i++)
    {
        auto freq = _mm256_set_pd(pfreq[3], pfreq[2], pfreq[1], pfreq[0]);
        auto tbg  = _mm256_set_pd(ptbg[3], ptbg[2], ptbg[1], ptbg[0]);
        auto tau  = _mm256_set_pd(ptau[3], ptau[2], ptau[1], ptau[0]);
        auto temp1 = _mm256_mul_pd(freq, vscale);
        tau = _mm256_mul_pd(tau, vm1);
        auto j_t = _mm256_mul_pd(temp1, vtexinv);
        auto j_tbg = _mm256_div_pd(temp1, tbg);
        tau = Sleef_expm1d4_u10avx2(tau);
        j_t = Sleef_expm1d4_u10avx2(j_t);
        j_tbg = Sleef_expm1d4_u10avx2(j_tbg);
        j_t = _mm256_div_pd(tau, j_t);
        j_tbg = _mm256_div_pd(tau, j_tbg);
        auto res = _mm256_sub_pd(j_tbg, j_t);
        _mm256_storeu_pd(ptb, res);
        pfreq += vecsize;
        ptbg += vecsize;
        ptau += vecsize;
        ptb += vecsize;
    }
    #elif defined(__ARM_NEON)
    const long vecsize = 2;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
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
        j_t = vdivq_f64(tau, j_t);
        j_tbg = vdivq_f64(tau, j_tbg);
        auto res = vsubq_f64(j_tbg, j_t);
        vst1q_f64(ptb, res);
        pfreq += vecsize;
        ptbg += vecsize;
        ptau += vecsize;
        ptb += vecsize;
    }
    #else
    const long remainder = len;
    #endif
    for (long i = 0; i < remainder; i++)
    {
        double temp1 = (*pfreq)*scale;
        double j_t = Sleef_expm1d1_u10(temp1*texinv);
        double j_tbg = Sleef_expm1d1_u10(temp1/(*ptbg));
        double tau = Sleef_expm1d1_u10(-(*ptau));
        j_t = tau/j_t;
        j_tbg = tau/j_tbg;
        *ptb = j_tbg - j_t;
        pfreq++;
        ptbg++;
        ptau++;
        ptb++;
    }
}