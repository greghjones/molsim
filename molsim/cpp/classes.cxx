#include <Python.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>
#include <cstdint>

#include <omp.h>

#include "pybind11/pytypes.h"

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#include "sleefinline_advsimd.hpp"
#endif

#include "sleefinline_purecfma_scalar.hpp"

#include "classes.hpp"
#include "constants.hpp"
#include "util.hpp"

#ifndef BLAS
template<typename T>
static inline void __attribute__((always_inline)) axpby(long n, double alpha, const T* x, long incx, double beta, T* y, long incy)
{
    if (incx == 1 && incy == 1)
    {
        if (beta == 0)
        {
            #pragma omp simd
            for (long i = 0; i < n; i++)
                y[i] = alpha*x[i];
        }
        else
        {
            #pragma omp simd
            for (long i = 0; i < n; i++)
                y[i] = alpha*x[i] + beta*y[i];
        }
    }
    else
    {
        const long imax = n*incx;
        if (beta == 0)
        {
            long j = 0;
            #pragma omp simd
            for (long i = 0; i < imax; i += incx )
            {
                y[j] = alpha*x[i];
                j += incy;
            }
        }
        else
        {
            long j = 0;
            #pragma omp simd
            for (long i = 0; i < imax; i += incx)
            {
                y[j] = alpha*x[i] + beta*y[j];
                j += incy;
            }
        }
    }
}

template<typename T>
static inline void __attribute__((always_inline)) scal(long n, double alpha, T* x, long incx)
{
    if (incx == 1)
        #pragma omp simd
        for (long i = 0; i < n; i++)
            x[i] *= alpha;
    else
    {
        const long maxi = n*incx;
        for (long i = 0; i < maxi; i += incx)
            x[i] *= alpha;
    }
}
#endif

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

    axpby(infreq.size(), scale, infreq.data(), 1, 0.0, outfreq.data(), 1);

    return outfreq;
}

std::pair<AlignedVector<double>,AlignedVector<double>> _apply_beam(const AlignedVector<double>& freq_array, const AlignedVector<double>& int_arr, double source_size, double dish_size)
{
    const double beam_size_factor = 206265.0 * 1.22 * (cm/1.0e6) / dish_size;
    const auto len = freq_array.size();

    assert(int_arr.size() == len);

    AlignedVector<double> result(len);
    AlignedVector<double> beam_dilution(len);

    const double* __restrict pf = freq_array.data();
    const double* __restrict pint = int_arr.data();
          double* __restrict pr = result.data();
          double* __restrict pbd = beam_dilution.data();

    #if defined(__AVX2__)
    const long vecsize = 4;
    const long unrollfac = 2;
    const long itlen = vecsize*unrollfac;
    const long maxit = len/itlen;
    const long remainder = len % itlen;
    auto fac = _mm256_set1_pd(beam_size_factor);
    auto ss2 = _mm256_set1_pd(source_size*source_size);
    for (long i = 0; i < maxit; i++)
    {
        auto f1 = _mm256_load_pd(pf);
        auto f2 = _mm256_load_pd(pf+vecsize);

        auto vi1 = _mm256_load_pd(pint);
        auto vi2 = _mm256_load_pd(pint+vecsize);

        auto bs1 = _mm256_div_pd(fac, f1);
        auto bs2 = _mm256_div_pd(fac, f2);

        auto bd1 = _mm256_fmadd_pd(bs1, bs1, ss2);
        auto bd2 = _mm256_fmadd_pd(bs2, bs2, ss2);

             bd1 = _mm256_div_pd(ss2, bd1);
             bd2 = _mm256_div_pd(ss2, bd2);

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
    #elif defined(__ARM_NEON)
    const long vecsize = 2;
    const long unrollfac = 2;
    const long itlen = vecsize*unrollfac;
    const long maxit = len/itlen;
    const long remainder = len % itlen;
    auto fac = vdupq_n_f64(beam_size_factor);
    auto ss2 = vdupq_n_f64(source_size*source_size);
    for (long i = 0; i < maxit; i++)
    {
        auto f1 = vld1q_f64(pf);
        auto f2 = vld1q_f64(pf+vecsize);

        auto vi1 = vld1q_f64(pint);
        auto vi2 = vld1q_f64(pint+vecsize);

        auto bs1 = vdivq_f64(fac, f1);
        auto bs2 = vdivq_f64(fac, f2);

        auto bd1 = vfmaq_f64(ss2, bs1, bs1);
        auto bd2 = vfmaq_f64(ss2, bs2, bs2);

             bd1 = vdivq_f64(ss2, bd1);
             bd2 = vdivq_f64(ss2, bd2);

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
    #else
    const long remainder = len;
    #pragma omp simd
    #endif
    for (long i = 0; i < remainder; i++)
    {
        const double beam_size = beam_size_factor / (*pf);
        const double ss2 = source_size*source_size;
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

template <typename T1, typename T2>
inline std::vector<char> _trim_arr_mask(const AlignedVector<T1>& arr, const AlignedVector<T2>& lls, const AlignedVector<T2>& uls, const AlignedVector<T2>& key_arr)
{
    assert(lls.size() == uls.size());
    assert(arr.size() == key_arr.size());

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

    long size = mask.size();

    for (long i = 0; i < size; i++)
        if (mask[i]) out.push_back(in[i]);

    return out;
}

template <typename T>
AlignedVector<long> _find_nearest(const AlignedVector<T>& searchfor, const AlignedVector<T>& in)
{
    AlignedVector<long> result(searchfor.size());
    for (long i = 0; i < searchfor.size(); i++)
    {
        auto val = searchfor[i];
        auto it = std::ranges::lower_bound(in, val);
        long j = std::distance(in.begin(), it);
        if (j > 0 && (it == in.end() || (abs(in[j-1] - val) < abs(in[j] - val)) ) )
            result[i] = j-1;
        else
            result[i] = j;
    }

    return result;
}

static inline void __attribute__((always_inline)) lowercase_str(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
}

Catalog::Catalog(const py::object& cat) :
    catid(0), molecule(), frequency(), freq_err(), logint(), sijmu(), sij(),
    aij(), elow(), eup(), glow(), gup(), measured(false)
{
    if (!cat.attr("catid").is_none())    catid = cat.attr("catid").cast<uint64_t>();
    if (!cat.attr("molecule").is_none()) molecule = cat.attr("molecule").cast<std::string>();

    if (!cat.attr("frequency").is_none())  frequency  = AlignedVector<double>(cat.attr("frequency").cast<py::array_t<double>>());
    if (!cat.attr("freq_err").is_none())   freq_err   = AlignedVector<double>(cat.attr("freq_err").cast<py::array_t<double>>());
    if (!cat.attr("logint").is_none())     logint     = AlignedVector<double>(cat.attr("logint").cast<py::array_t<double>>());
    if (!cat.attr("sijmu").is_none())      sijmu      = AlignedVector<double>(cat.attr("sijmu").cast<py::array_t<double>>());
    if (!cat.attr("sij").is_none())        sij        = AlignedVector<double>(cat.attr("sij").cast<py::array_t<double>>());
    if (!cat.attr("aij").is_none())        aij        = AlignedVector<double>(cat.attr("aij").cast<py::array_t<double>>());
    if (!cat.attr("elow").is_none())       elow       = AlignedVector<double>(cat.attr("elow").cast<py::array_t<double>>());
    if (!cat.attr("eup").is_none())        eup        = AlignedVector<double>(cat.attr("eup").cast<py::array_t<double>>());
    if (!cat.attr("glow").is_none())       glow       = AlignedVector<int>(cat.attr("glow").cast<py::array_t<int>>());
    if (!cat.attr("gup").is_none())        gup        = AlignedVector<int>(cat.attr("gup").cast<py::array_t<int>>());
    if (!cat.attr("measured").is_none())   measured = cat.attr("measured").cast<bool>();
}

Molecule::Molecule(const py::object& mol_py) :
    qpart(mol_py.attr("qpart").cast<py::object>(), this),
    catalog(mol_py.attr("catalog").cast<py::object>()),
    level_degeneracies(),
    level_energies()
{
    if(!mol_py.attr("levels").is_none())
    {
        const py::list& levellist = mol_py.attr("levels").cast<py::list>();
        auto nlevels = levellist.size();
        level_degeneracies.reserve(nlevels);
        level_energies.reserve(nlevels);
        for (auto& level : levellist)
        {
            level_degeneracies.push_back(level.attr("g").cast<double>());
            level_energies.push_back(level.attr("energy").cast<double>());
        }
    }
}

PartitionFunction::PartitionFunction(const py::object& qpart, Molecule* mol)
{
    std::string flag_s = qpart.attr("flag").cast<std::string>();
    lowercase_str(flag_s);
    if (flag_s == "interpolation")
    {
        flag = interpolation;
        auto t = qpart.attr("temps").cast<py::array_t<double>>();

        sigma = 1.0;

        temps = AlignedVector<double>(t);
        vals = AlignedVector<double>(qpart.attr("vals").cast<py::array_t<double>>());
    }
    else if (flag_s == "counting")
    {
        if (!mol) ERROR("Counting selected for partition function type, but no molecule attached!");
        sigma = qpart.attr("sigma").is_none() ? 1.0 : qpart.attr("sigma").cast<double>();
        flag = counting;
        parent_mol = mol;
    }
    else
        ERROR("Only interpolation implemented for partition function! Selected type is: {}", flag_s);


    if (!qpart.attr("vib_states").is_none())
    {
        ERROR("Vibrational partition functions not yet implemented!");
        return;
    }
}

double PartitionFunction::qrot(double Tex)
{
    if (flag == interpolation)
{
    int i = 0;
    while(Tex > temps[i] && i < temps.size()) i++;

    if (i == 0) return vals[0];
    if (i == temps.size()) return *vals.end();

    const double m = (vals[i] - vals[i-1])/(temps[i] - temps[i-1]);
    double result = vals[i-1] + m*(Tex - temps[i-1]);
    return result;
    }
    else if (flag == counting) return qrot_counting(Tex);
    else ERROR("Only \"interpolation\" and \"counting\" supported for rotational partition function.");
}

double PartitionFunction::qrot_counting(double Tex)
{
    const Molecule& mol = *parent_mol;
    auto& e = mol.level_energies; // in K
    auto& g = mol.level_degeneracies;

    auto len = e.size();
    assert(g.size() == len);

    double Tinv = -1.0/Tex;

    double result = 0.0;

    #pragma omp simd
    for (long i = 0; i < len; i++)
        result += g[i]*std::exp(e[i]*Tinv);

    return result/sigma;
}

Spectrum::Spectrum() :
    freq0(), frequency(), Tb(), Iv(), Tbg(), Ibg(), tau(),
    tau_profile(), freq_profile(), int_profile(), Tbg_profile() { }

Spectrum::Spectrum(const py::object& spec) : Spectrum()
{
    if (!spec.attr("freq0").is_none())         freq0         = AlignedVector<double>(spec.attr("freq0").cast<py::array_t<double>>());
    if (!spec.attr("frequency").is_none())     frequency     = AlignedVector<double>(spec.attr("frequency").cast<py::array_t<double>>());
    if (!spec.attr("Tb").is_none())            Tb            = AlignedVector<double>(spec.attr("Tb").cast<py::array_t<double>>());
    if (!spec.attr("Iv").is_none())            Iv            = AlignedVector<double>(spec.attr("Iv").cast<py::array_t<double>>());
    if (!spec.attr("Tbg").is_none())           Tbg           = AlignedVector<double>(spec.attr("Tbg").cast<py::array_t<double>>());
    if (!spec.attr("Ibg").is_none())           Ibg           = AlignedVector<double>(spec.attr("Ibg").cast<py::array_t<double>>());
    if (!spec.attr("tau").is_none())           tau           = AlignedVector<double>(spec.attr("tau").cast<py::array_t<double>>());
    if (!spec.attr("tau_profile").is_none())   tau_profile   = AlignedVector<double>(spec.attr("tau_profile").cast<py::array_t<double>>());
    if (!spec.attr("freq_profile").is_none())  freq_profile  = AlignedVector<double>(spec.attr("freq_profile").cast<py::array_t<double>>());
    if (!spec.attr("int_profile").is_none())   int_profile   = AlignedVector<double>(spec.attr("int_profile").cast<py::array_t<double>>());
    if (!spec.attr("Tbg_profile").is_none())   Tbg_profile   = AlignedVector<double>(spec.attr("Tbg_profile").cast<py::array_t<double>>());
}

Continuum::Continuum() :
    cont_file(), type(thermal), params(2.7), freqs(), temps(), fluxes(), notes() { }

Continuum::Continuum(const py::object& cont) : Continuum()
{
    if (!cont.attr("cont_file").is_none()) cont_file = cont.attr("cont_file").cast<std::string>();

    std::string type_s = cont.attr("type").cast<std::string>();
    lowercase_str(type_s);
    if (type_s != "thermal")
    {
        ERROR("Non-thermal background not yet implemented!");
        return;
    }

    params = cont.attr("params").cast<double>();
    if (!cont.attr("freqs").is_none())  freqs  = AlignedVector<double>(cont.attr("freqs").cast<py::array_t<double>>());
    if (!cont.attr("temps").is_none())  temps  = AlignedVector<double>(cont.attr("temps").cast<py::array_t<double>>());
    if (!cont.attr("fluxes").is_none()) fluxes = AlignedVector<double>(cont.attr("fluxes").cast<py::array_t<double>>());
}

void Continuum::Tbg(const AlignedVector<double>& freq, AlignedVector<double>& tbg)
{
    if (type == thermal) 
    {
        if (tbg.size() != freq.size())
            tbg.resize(freq.size());
        for (auto& v : tbg) v = params;
    }
    else
        ERROR("Only thermal continuum supported.");
}

void Continuum::Ibg(const AlignedVector<double>& freq, double tbg, AlignedVector<double>& ibg)
{
    const auto len = freq.size();
    if (ibg.size() != len) ibg.resize(len);

    const double f = 2.0e26 / (cm * cm);

    const double* __restrict pfreq = freq.data();
          double* __restrict pibg  = ibg.data();

    #if defined(__AVX2__)
    const int vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len % vecsize;
    const auto vf = _mm256_set1_pd(f);
    const auto vmhz = _mm256_set1_pd(1.0e6);
    const auto vh = _mm256_set1_pd(h);
    const auto vk = _mm256_set1_pd(k);
    const auto vt = _mm256_set1_pd(tbg);
    for (long i = 0; i < maxit; i++)
    {
        auto v0 = _mm256_load_pd(pfreq);
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
        pibg  += vecsize;
    }
    #elif defined(__ARM_NEON)
    const int vecsize = 2;
    const auto maxit = len/vecsize;
    const auto remainder = len % vecsize;
    const auto vf = vdupq_n_f64(f);
    const auto vmhz = vdupq_n_f64(1.0e6);
    const auto vh = vdupq_n_f64(h);
    // const auto vk = vdupq_n_f64(k);
    // const auto vt = vdupq_n_f64(tbg);
    const auto v3 = vdupq_n_f64(k*tbg);
    for (long i = 0; i < maxit; i++)
    {
        auto v0 = vld1q_f64(pfreq);
        auto v1 = vmulq_f64(v0, vmhz);
        auto v2 = vmulq_f64(v1, vh);
        auto v4 = vdivq_f64(v2, v3);
        auto v5 = vmulq_f64(v1, v2);
             v5 = vmulq_f64(v1, v5);
        auto v6 = Sleef_expm1d2_u10advsimd(v4);
        auto v7 = vmulq_f64(vf, v5);
        auto res = vdivq_f64(v7, v6);
        vst1q_f64(pibg, res);
        pfreq += vecsize;
        pibg += vecsize;
    }
    #else
    const auto remainder = len;
    #endif
    const double s3 = tbg*k;
    for (long i = 0; i < remainder; i++)
    {
        double v1 = (*pfreq)*1.0e6;
        double v2 = v1*h;
        double v4 = v2/s3;
        double v5 = v1*v1*v2;
        double v6 = Sleef_expm1d1_u10purecfma(v4);
        double v7 = f*v5;
        *pibg = v6*v7;
        pfreq++;
        pibg++;
    }
}

void Continuum::Ibg(const AlignedVector<double>& freq, const AlignedVector<double>& tbg, AlignedVector<double>& ibg)
{
    const auto len = freq.size();
    if (ibg.size() != len) ibg.resize(len);

    const double f = 2.0e26 / (cm * cm);

    const double* __restrict ptbg = tbg.data();
    
    const double* __restrict pfreq = freq.data();
          double* __restrict pibg  = ibg.data();

    #if defined(__AVX2__)
    const int vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len % vecsize;
    const auto vf = _mm256_set1_pd(f);
    const auto vmhz = _mm256_set1_pd(1.0e6);
    const auto vh = _mm256_set1_pd(h);
    const auto vk = _mm256_set1_pd(k);
    for (long i = 0; i < maxit; i++)
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
    #elif defined(__ARM_NEON)
    const int vecsize = 2;
    const auto maxit = len/vecsize;
    const auto remainder = len % vecsize;
    const auto vf = vdupq_n_f64(f);
    const auto vmhz = vdupq_n_f64(1.0e6);
    const auto vh = vdupq_n_f64(h);
    const auto vk = vdupq_n_f64(k);
    for (long i = 0; i < maxit; i++)
    {
        auto v0 = vld1q_f64(pfreq);
        auto vt = vld1q_f64(ptbg);
        auto v1 = vmulq_f64(v0, vmhz);
        auto v2 = vmulq_f64(v1, vh);
        auto v3 = vmulq_f64(vt, vk);
        auto v4 = vdivq_f64(v2, v3);
        auto v5 = vmulq_f64(v1, v2);
             v5 = vmulq_f64(v1, v5);
        auto v6 = Sleef_expm1d2_u10advsimd(v4);
        auto v7 = vmulq_f64(vf, v5);
        auto res = vdivq_f64(v7, v6);
        vst1q_f64(pibg, res);
        pfreq += vecsize;
        ptbg += vecsize;
        pibg += vecsize;
    }
    #else
    const auto remainder = len;
    #pragma omp simd
    #endif
    for (long i = 0; i < remainder; i++)
    {
        double v1 = (*pfreq)*1.0e6;
        double v2 = v1*h;
        double v3 = (*ptbg)*k;
        double v4 = v2/v3;
        double v5 = v1*v1*v2;
        double v6 = Sleef_expm1d1_u10purecfma(v4);
        double v7 = f*v5;
        *pibg = v6*v7;
        pfreq++;
        ptbg++;
        pibg++;
    }
}

Source::Source() :
    name(""), velocity(0.0), size(1.0e20), solid_angle(), continuum(),
    column(1.0e13), Tex(300.0), Tkin(), dV(3.0), id(0), notes("") { }

Source::Source(const py::object& source) : Source()
{
    if(!source.attr("name").is_none()) name = source.attr("name").cast<std::string>();
    velocity = source.attr("velocity").cast<double>();
    size = source.attr("size").cast<double>();
    if(!source.attr("solid_angle").is_none()) solid_angle = source.attr("solid_angle").cast<double>();
    column = source.attr("column").cast<double>();
    Tex = source.attr("Tex").cast<double>();
    if(!source.attr("Tkin").is_none()) Tkin = source.attr("Tkin").cast<double>();
    dV = source.attr("dV").cast<double>();
    if(!source.attr("id").is_none()) id = source.attr("id").cast<uint64_t>();
    if(!source.attr("notes").is_none()) notes = source.attr("notes").cast<std::string>();
}

Observatory::Observatory() :
    name(""), id(0), sd(true), array(false), dish(100.0), synth_beam{1.0, 1.0},
    eta(), eta_type(constant), eta_params(), atmo() { }

Observatory::Observatory(const py::object& obs) : Observatory()
{
    if(!obs.attr("name").is_none()) name = obs.attr("name").cast<std::string>();
    if(!obs.attr("id").is_none()) id = obs.attr("id").cast<uint64_t>();
    sd = obs.attr("sd").cast<bool>();
    array = obs.attr("array").cast<bool>();
    dish = obs.attr("dish").cast<double>();
    auto synth_beam_py = static_cast<py::list>(obs.attr("synth_beam"));
    synth_beam = {synth_beam_py[0].cast<double>(), synth_beam_py[1].cast<double>()};
    if (!obs.attr("eta").is_none()) eta = AlignedVector<double>(obs.attr("eta").cast<py::array_t<double>>());
    
    std::string eta_type_s = obs.attr("eta_type").cast<std::string>();
    lowercase_str(eta_type_s);
    if (eta_type_s != "constant")
    {
        ERROR("Invalid eta_type for Observatory!");
        return;
    }

    if (!obs.attr("atmo").is_none()) atmo = AlignedVector<double>(obs.attr("atmo").cast<py::array_t<double>>());
}

Observation::Observation() :
    vlsr(0.0), spectrum(), observatory(), id(0),
    notes("") { }

Observation::Observation(const py::object& obs) : Observation()
{
    if(!obs.attr("spectrum").is_none())    spectrum = Spectrum(obs.attr("spectrum"));
    if(!obs.attr("observatory").is_none()) observatory = Observatory(obs.attr("observatory"));
    if(!obs.attr("vlsr").is_none())        vlsr = obs.attr("vlsr").cast<double>();
    if(!obs.attr("id").is_none())          id = obs.attr("id").cast<uint64_t>();
    if(!obs.attr("notes").is_none())       notes = obs.attr("notes").cast<std::string>();
}

Simulation::Simulation(const py::object& spectrum_py,
                       const py::object& observation_py,
                       const py::object& source_py,
                       const py::array_t<double>& ll_py,
                       const py::array_t<double>& ul_py,
                       const std::string& line_profile_,
                       double sim_width_,
                       double res_,
                       const py::object& mol_py,
                             std::string units_,
                       const std::string& notes_,
                       bool use_obs_,
                       bool add_noise_,
                       double noise_,
                       double tau_threshold_,
                       double eup_threshold_) :
    spectrum(), observation(observation_py.is_none() ? std::optional<Observation>() : observation_py), source(source_py),
    ll(ll_py), ul(ul_py), sim_width(sim_width_), res(res_), mol(mol_py), notes(notes_), use_obs(use_obs_), add_noise_flag(add_noise_),
    noise(noise_), tau_threshold(tau_threshold_), eup_threshold(eup_threshold_), beam_dilution(), l_idxs(), u_idxs()
{
    if (!spectrum_py.is_none()) ERROR("Attaching initial spectrum not yet implemented.");
    if(add_noise_flag)          ERROR("Adding noise not yet implemented.");

    set_line_profile(line_profile_);

    lowercase_str(units_);
    if (units_ == "k")
        units = K;
    else if (units_ == "mk")
        units = mK;
    else if (units_ == "jy/beam")
        units = Jy_beam;
    else
        ERROR("Invalid unit type for Simulation!");

    set_arrays();
    if (spectrum.frequency.empty())
    {
        spectrum.freq_profile.clear();
        spectrum.int_profile.clear(); 
    }
    apply_voffset();
    calc_tau();
    calc_bg();
    // calc_Iv(); // implement later, not used in simple workflows it seems
    if (source.continuum.type == Continuum::thermal)
        calc_Tb(spectrum.frequency, spectrum.tau, source.continuum.params, source.Tex, spectrum.Tb);
    else
    calc_Tb(spectrum.frequency, spectrum.tau, spectrum.Tbg, source.Tex, spectrum.Tb);
    make_lines();
    beam_correct();
    set_units();
    add_noise();
}

void Simulation::set_line_profile(std::string label)
{
    lowercase_str(label);

    if (label == "gaussian") line_profile = Gaussian;
    else if (label.empty()) line_profile = NoLineProfile;
    else
        ERROR("Invalid line_profile type in Simulation!\n");
};

void Simulation::set_units()
{
    if (units == K)
        return;

    if (units == mK)
    {
        if (line_profile == Gaussian)
            scal(spectrum.int_profile.size(), 1000.0, spectrum.int_profile.data(), 1);
        scal(spectrum.Tb.size(), 1000.0, spectrum.Tb.data(), 1);
        return;
    }

    if (units == Jy_beam)
    {
        if (!observation.has_value() || (!observation->observatory.has_value()))
            ERROR("Missing observation data for synth_beam!");

        // const double omega = (observation->observatory->synth_beam.first) * (observation->observatory->synth_beam.second);
        ERROR("Unimplemented until I figure out meaning of magic numbers.");
    }
};

void Simulation::update()
{
    set_arrays();
    apply_voffset();
    calc_tau();
    calc_bg();
    // calc_Iv();
    calc_Tb(spectrum.frequency, spectrum.tau, spectrum.Tbg, source.Tex, spectrum.Tb);
    make_lines();
    beam_correct();
    set_units();
    // add_noise();
}

void Simulation::set_arrays()
{
    const double f = 1.0 - source.velocity/ckm;
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
    const double prefactor = std::sqrt(std::log(2.0)) * cm*cm*cm * (source.column * 100*100) 
                             / (4*std::pow(std::numbers::pi,1.5) * 1.0e18 * source.dV * 1000.0 * mol.q(source.Tex));

    const auto len = spectrum.frequency.size();
    spectrum.tau.resize(len);

    for (auto& element : spectrum.tau) element = 0.0;

    const    int* __restrict pgup = gup.data();
    const double* __restrict paij = aij.data();
    const double* __restrict peup = eup.data();
    const double* __restrict pfreq = spectrum.frequency.data();
          double* __restrict ptau = spectrum.tau.data();


    #if defined(__AVX2__)
    const long vecsize = 4;
    const long maxit = len/vecsize;
    const long remainder = len - maxit*vecsize;
    auto vtexinv    = _mm256_set1_pd(texinv);
    auto vboltzmann = _mm256_set1_pd(boltzmannscale);
    auto vprefactor = _mm256_set1_pd(prefactor);
    for (long i = 0; i < maxit; i++)
    {
        auto exp1 = _mm256_load_pd(peup);
        exp1 = _mm256_mul_pd(vtexinv, exp1);
        exp1 = Sleef_expd4_u10avx2(exp1);
        auto f = _mm256_load_pd(pfreq);
        auto exp2 = _mm256_mul_pd(vboltzmann, f);
        exp2 = Sleef_expd4_u10avx2(exp2);
        auto finv = _mm256_mul_pd(f, f);
        finv = _mm256_mul_pd(finv, f);
        finv = _mm256_div_pd(vprefactor, finv);
        auto gupint = _mm_load_epi32(pgup);
        auto gup = _mm256_cvtepi32_pd(gupint);
        auto res = _mm256_load_pd(paij);
        res = _mm256_mul_pd(res, gup);
        exp1 = _mm256_mul_pd(exp1, exp2);
        res = _mm256_mul_pd(res, finv);
        res = _mm256_mul_pd(res, exp1);
        _mm256_store_pd(ptau, res);
        pgup  += vecsize;
        peup  += vecsize;
        paij  += vecsize;
        pfreq += vecsize;
        ptau  += vecsize;
    }
    #elif defined(__ARM_NEON)
    const long vecsize = 2;
    const long maxit = len/vecsize;
    const long remainder = len % vecsize;
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

void Simulation::calc_bg()
{
    source.continuum.Tbg(spectrum.frequency, spectrum.Tbg);
    if (source.continuum.type == Continuum::thermal)
        source.continuum.Ibg(spectrum.frequency, source.continuum.params, spectrum.Ibg);
    else
        source.continuum.Ibg(spectrum.frequency, spectrum.Tbg, spectrum.Ibg);
}

void Simulation::calc_Tb(const AlignedVector<double> frequency,
                         const AlignedVector<double>& tau,
                         const AlignedVector<double>& Tbg,
                         double Tex,
                               AlignedVector<double>& Tb)
{
    const auto len = frequency.size();
    const double texinv = 1.0/Tex;
    const double scale = h*1.0e6/k;

    assert(Tbg.size() == len);
    assert(tau.size() == len);

    Tb.resize(len);

    const double* __restrict pfreq = frequency.data();
    const double* __restrict ptbg = Tbg.data();
    const double* __restrict ptau = tau.data();
          double* __restrict ptb = Tb.data();

    #if defined(__AVX2__)
    const long vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
    const auto vtexinv = _mm256_set1_pd(texinv);
    const auto vscale = _mm256_set1_pd(scale);
    const auto vm1 = _mm256_set1_pd(-1.0);
    for (long i = 0; i < maxit; i++)
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
        res = _mm256_mul_pd(tau, temp1);
        _mm256_store_pd(ptb, res);
        pfreq += vecsize;
        ptbg += vecsize;
        ptau += vecsize;
        ptb += vecsize;
    }
    #elif defined(__ARM_NEON)
    const long vecsize = 2;
    const long unrollfactor = 1;
    const long itlen = vecsize*unrollfactor;
    const auto maxit = len/itlen;
    const auto remainder = len % itlen;
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
        ptbg += itlen;
        ptau += itlen;
        ptb += itlen;
    }
    #else
    const long remainder = len;
    #endif
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

void Simulation::calc_Tb(const AlignedVector<double> frequency,
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
    const long offset = len / nthreads;
    const int tid = omp_get_thread_num();
    const long localoffset = offset*tid;
    const double* __restrict pfreq = frequency.data()+localoffset;
    const double* __restrict ptau = tau.data()+localoffset;
          double* __restrict ptb = Tb.data()+localoffset;
          

    #if defined(__AVX2__)
    const long vecsize = 4;
    const auto maxit = len/vecsize;
    const auto remainder = len - maxit*vecsize;
    const auto vtexinv = _mm256_set1_pd(texinv);
    const auto vscale = _mm256_set1_pd(scale);
    const auto vm1 = _mm256_set1_pd(-1.0);
    const auto tbg = _mm256_set1_pd(Tbg);
    for (long i = 0; i < maxit; i++)
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
        pfreq += vecsize;
        ptau += vecsize;
        ptb += vecsize;
    }
    #elif defined(__ARM_NEON)
    const long vecsize = 2;
    const long unrollfactor = 4;
    const long itlen = vecsize*unrollfactor;
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
    #else
    const long remainder = len;
    #endif
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

void Simulation::make_lines()
{
    if (line_profile == NoLineProfile)
        return;

    if (line_profile == Gaussian)
    {
        if(use_obs)
        {
            if (!observation.has_value())
            {
                ERROR("No observation data, but set use_obs!");
                return;
            }
            spectrum.freq_profile = observation->spectrum.frequency;
            if (l_idxs.empty())
            {
                double windowfactor = sim_width*source.dV/ckm;
                AlignedVector<double> lls_raw(spectrum.frequency.size());
                AlignedVector<double> uls_raw(spectrum.frequency.size());
                axpby(spectrum.frequency.size(), 1.0-windowfactor, spectrum.frequency.data(), 1, 0.0, lls_raw.data(), 1);
                axpby(spectrum.frequency.size(), 1.0+windowfactor, spectrum.frequency.data(), 1, 0.0, uls_raw.data(), 1);
                l_idxs = _find_nearest(lls_raw, spectrum.freq_profile);
                u_idxs = _find_nearest(uls_raw, spectrum.freq_profile);
            }
        }
        else
        {
            double windowfactor = sim_width*source.dV/ckm;
            AlignedVector<double> lls_raw(spectrum.frequency.size());
            AlignedVector<double> uls_raw(spectrum.frequency.size());
            axpby(spectrum.frequency.size(), 1.0-windowfactor, spectrum.frequency.data(), 1, 0.0, lls_raw.data(), 1);
            axpby(spectrum.frequency.size(), 1.0+windowfactor, spectrum.frequency.data(), 1, 0.0, uls_raw.data(), 1);
            AlignedVector<double> ll_trim = {lls_raw[0]};
            AlignedVector<double> ul_trim = {uls_raw[0]};
            for (long i = 1; i < lls_raw.size(); i++)
            {
                if (lls_raw[i] < ul_trim.back())
                    ul_trim.back() = uls_raw[i];
                else
                {
                    ll_trim.push_back(lls_raw[i]);
                    ul_trim.push_back(uls_raw[i]);
                }
            }

            spectrum.freq_profile.clear();

            for (long i = 0; i < ll_trim.size(); i++)
                for (double v = ll_trim[i]; v < ul_trim[i]; v += res)
                    spectrum.freq_profile.push_back(v);

            l_idxs = _find_nearest(lls_raw, spectrum.freq_profile);
            u_idxs = _find_nearest(uls_raw, spectrum.freq_profile);
        }

        source.continuum.Tbg(spectrum.freq_profile, spectrum.Tbg_profile);

        // if (doonce)
        // {
            make_gaussians(spectrum.frequency, spectrum.tau, l_idxs, u_idxs, source.dV, spectrum.freq_profile, spectrum.tau_profile);
        //     doonce = false;
        // }

        // if (doonce)
        // {
            // Update int_profile, separate functions for constant thermal background
            if (source.continuum.type == Continuum::thermal)
                calc_Tb(spectrum.freq_profile, spectrum.tau_profile, source.continuum.params, source.Tex, spectrum.int_profile);
            else
                calc_Tb(spectrum.freq_profile, spectrum.tau_profile, spectrum.Tbg_profile, source.Tex, spectrum.int_profile);

        //     doonce = false;
        // }

        if(!use_obs)
        {
            l_idxs.clear();
            u_idxs.clear();
        }
    }
}

void Simulation::make_gaussians(const AlignedVector<double>& centers, const AlignedVector<double>& int0s,
                               const AlignedVector<long>& lls, const AlignedVector<long>& uls, double dV,
                               const AlignedVector<double>& x, AlignedVector<double>& y)
{
    auto npeaks = centers.size();

    y.resize(x.size());
    for (auto& element : y) element = 0.0;

    assert(int0s.size() == npeaks);
    assert(lls.size() == npeaks);
    assert(uls.size() == npeaks);

    const double scale1 = 2*(dV/ckm/2.35482)*(dV/ckm/2.35482);

    // const int nthreads = omp_get_num_threads();

    // std::vector<AlignedVector<double>> buffer(nthreads, AlignedVector<double>(y.size()));

    // #pragma omp parallel
    // {
    //     const int tid = omp_get_thread_num();
        for (long n = 0; n < npeaks; n++)
        {
            // if ( (n % nthreads) != tid) continue;
            const double center = centers[n];
            const double int0 = int0s[n];
            const double scale2 = -1.0 / (scale1*center*center);
            const double* __restrict px = x.data() + lls[n];
                  double* __restrict py = y.data() + lls[n];
                //   double* __restrict py = buffer[tid].data() + lls[n];
            const auto len = uls[n] - lls[n];

            #if defined(__AVX2__)
            const long vecsize = 4;
            const long maxit = len/vecsize;
            const long remainder = len % vecsize;
            auto vs = _mm256_set1_pd(scale2);
            auto vc = _mm256_set1_pd(center);
            auto va = _mm256_set1_pd(int0);
            for (long i = 0; i < maxit; i++)
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
            #elif defined(__ARM_NEON)
            const long vecsize = 2;
            const long unrollfac = 4;
            const long itlen = vecsize*unrollfac;
            const long maxit = len/itlen;
            const long remainder = len % itlen;
            auto vs = vdupq_n_f64(scale2);
            auto vc = vdupq_n_f64(center);
            auto va = vdupq_n_f64(int0);
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
            #else
            const long remainder = len;
            #pragma omp simd
            #endif
            for (long i = 0; i < remainder; i++)
            {
                double v = *px - center;
                v = Sleef_expd1_u10purecfma(scale2*v*v);
                *py += int0*v;
                px++;
                py++;
            }
        }
    // }

    // for (long i = 0; i < y.size(); i++)
    // for (int j = 0; j < nthreads; j++)
    //     y[i] += buffer[j][i];
    
}

void Simulation::beam_correct()
{
    if (!(observation.has_value()) || !(observation->observatory.has_value()) || !(observation->observatory->sd))
        return;

    std::tie(spectrum.Tb, beam_dilution) = _apply_beam(spectrum.frequency, spectrum.Tb, source.size, observation->observatory->dish);
    std::tie(spectrum.int_profile, std::ignore) = _apply_beam(spectrum.freq_profile, spectrum.int_profile, source.size, observation->observatory->dish);
}
