#include <Python.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#else
    // Fallback for sequential execution
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
#endif

#include "pybind11/pytypes.h"

#if defined(__AVX2__)
#include <emmintrin.h>
#include <immintrin.h>
#include "sleefinline_avx2.h"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#include "sleefinline_advsimd.hpp"
#endif

#include "sleefinline_purecfma_scalar.hpp"

#include "classes.hpp"
#include "constants.hpp"
#include "util.hpp"
#include "functional.hpp"


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
AlignedVector<ssize_t> _find_nearest(const AlignedVector<T>& searchfor, const AlignedVector<T>& in)
{
    AlignedVector<ssize_t> result(searchfor.size());
    for (ssize_t i = 0; i < searchfor.size(); i++)
    {
        auto val = searchfor[i];
        auto it = std::ranges::lower_bound(in, val);
        ssize_t j = std::distance(in.begin(), it);
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
    const long vecsize = 4;
    const long unrollfactor = 1;
    const long itlen = vecsize*unrollfactor;
    const auto maxit = len/itlen;
    const auto remainder = len % itlen;
    const auto vf = _mm256_set1_pd(f);
    const auto vmhz = _mm256_set1_pd(1.0e6);
    const auto vh = _mm256_set1_pd(h);
    const auto v3 = _mm256_set1_pd(k*tbg);
    for (long i = 0; i < maxit; i++)
    {
        auto v0 = _mm256_load_pd(pfreq);
        auto v1 = _mm256_mul_pd(v0, vmhz);
        auto v2 = _mm256_mul_pd(v1, vh);
        auto v4 = _mm256_div_pd(v2, v3);
        auto v5 = _mm256_mul_pd(v1, v2);
             v5 = _mm256_mul_pd(v1, v5);
        auto v6 = Sleef_expm1d4_u10avx2(v4);
        auto v7 = _mm256_mul_pd(vf, v5);
        auto res = _mm256_div_pd(v7, v6);
        _mm256_store_pd(pibg, res);
        pfreq += itlen;
        pibg  += itlen;
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
        *pibg = v7/v6;
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
        *pibg = v7/v6;
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
        molsim::functional::calc_Tb(spectrum.frequency, spectrum.tau, source.continuum.params, source.Tex, spectrum.Tb);
    else
        molsim::functional::calc_Tb(spectrum.frequency, spectrum.tau, spectrum.Tbg, source.Tex, spectrum.Tb);
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
    molsim::functional::calc_Tb(spectrum.frequency, spectrum.tau, spectrum.Tbg, source.Tex, spectrum.Tb);
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
    spectrum.frequency = molsim::functional::apply_vlsr(spectrum.freq0, source.velocity);
}

void Simulation::calc_tau()
{
    molsim::functional::calc_tau(aij, gup, eup,
                                 spectrum.frequency, source.column, source.Tex, source.dV,
                                 mol.q(source.Tex), spectrum.tau);
}

void Simulation::calc_bg()
{
    source.continuum.Tbg(spectrum.frequency, spectrum.Tbg);
    if (source.continuum.type == Continuum::thermal)
        source.continuum.Ibg(spectrum.frequency, source.continuum.params, spectrum.Ibg);
    else
        source.continuum.Ibg(spectrum.frequency, spectrum.Tbg, spectrum.Ibg);
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

        molsim::functional::make_gaussians(spectrum.frequency, spectrum.tau, l_idxs, u_idxs, source.dV, spectrum.freq_profile, spectrum.tau_profile);

        if (source.continuum.type == Continuum::thermal)
            molsim::functional::calc_Tb(spectrum.freq_profile, spectrum.tau_profile, source.continuum.params, source.Tex, spectrum.int_profile);
        else
            molsim::functional::calc_Tb(spectrum.freq_profile, spectrum.tau_profile, spectrum.Tbg_profile, source.Tex, spectrum.int_profile);
    
        if(!use_obs)
        {
            l_idxs.clear();
            u_idxs.clear();
        }
    }
}

void Simulation::beam_correct()
{
    if (!(observation.has_value()) || !(observation->observatory.has_value()) || !(observation->observatory->sd))
        return;

    std::tie(spectrum.Tb, beam_dilution)        = molsim::functional::apply_beam(spectrum.frequency, spectrum.Tb, source.size, observation->observatory->dish);
    std::tie(spectrum.int_profile, std::ignore) = molsim::functional::apply_beam(spectrum.freq_profile, spectrum.int_profile, source.size, observation->observatory->dish);
}
