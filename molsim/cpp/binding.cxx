#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <variant>

#include "classes.hpp"
#include "molsim/cpp/functional.hpp"
#include "molsim/cpp/util.hpp"
#include "molsim/cpp/constants.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
using nparray = py::array_t<T, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(molsim_cpp, m, py::mod_gil_not_used()) {
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<const py::object&,
                      const py::object&,
                      const py::object&, 
                      const py::array_t<double>&,
                      const py::array_t<double>&,
                      const std::string&,
                      double,
                      double,
                      const py::object&,
                      std::string,
                      const std::string&,
                      bool,
                      bool,
                      double,
                      double,
                      double>(),
                      "spectrum"_a = py::none(),
                      "observation"_a = py::none(),
                      "source"_a = py::none(),
                      "ll"_a = py::array_t<double>(),
                      "ul"_a = py::array_t<double>(),
                      "line_profile"_a = "gaussian",
                      "sim_width"_a = 10.0,
                      "res"_a = 0.010,
                      "mol"_a = py::none(),
                      "units"_a = "K",
                      "notes"_a = "",
                      "use_obs"_a = false, 
                      "add_noise"_a = false,
                      "noise"_a = 0.0,
                      "tau_threshold"_a = 0.0,
                      "eup_threshold"_a = 0.0)
        .def("update", &Simulation::update)
        .def_property("source", &Simulation::get_source, &Simulation::set_source)
        .def("_apply_voffset", &Simulation::apply_voffset)
        .def("_calc_tau", &Simulation::calc_tau)
        .def("_make_lines", &Simulation::make_lines)
        .def("_beam_correct", &Simulation::beam_correct)
        .def_property("spectrum", [](const Simulation& s){ return s.spectrum; }, nullptr);

    py::class_<Spectrum>(m, "Spectrum")
        .def_property("freq0",        [](const Spectrum& s){ return py::array_t<double>(s.freq0.size(),        s.freq0.data());         }, nullptr)
        .def_property("frequency",    [](const Spectrum& s){ return py::array_t<double>(s.frequency.size(),    s.frequency.data());     }, nullptr)
        .def_property("tau",          [](const Spectrum& s){ return py::array_t<double>(s.tau.size(),          s.tau.data());           }, nullptr)
        .def_property("Ibg",          [](const Spectrum& s){ return py::array_t<double>(s.Ibg.size(),          s.Ibg.data());           }, nullptr)
        .def_property("Tbg",          [](const Spectrum& s){ return py::array_t<double>(s.Tbg.size(),          s.Tbg.data());           }, nullptr)
        .def_property("Tb",           [](const Spectrum& s){ return py::array_t<double>(s.Tb.size(),           s.Tb.data());            }, nullptr)
        .def_property("freq_profile", [](const Spectrum& s){ return py::array_t<double>(s.freq_profile.size(), s.freq_profile.data());  }, nullptr)
        .def_property("Tbg_profile",  [](const Spectrum& s){ return py::array_t<double>(s.Tbg_profile.size(),  s.Tbg_profile.data());   }, nullptr)
        .def_property("tau_profile",  [](const Spectrum& s){ return py::array_t<double>(s.tau_profile.size(),  s.tau_profile.data());   }, nullptr)
        .def_property("int_profile",  [](const Spectrum& s){ return py::array_t<double>(s.int_profile.size(),  s.int_profile.data());   }, nullptr);
    
    auto functional = m.def_submodule("functional", "Optimized pure functions for spectral simulation.");

    functional.def("calc_tau",
    [](nparray<double> aij,
       nparray<int> gup,
       nparray<double> eup,
       nparray<double> frequencies,
       double columndensity,
       double Tex,
       double dV,
       double q) -> py::array_t<double>
    {
        AlignedVector<double> tau(frequencies.size());
        molsim::functional::calc_tau(AlignedVector<double>(aij),
                                     AlignedVector<int>(gup),
                                     AlignedVector<double>(eup),
                                     AlignedVector<double>(frequencies),
                                     columndensity, Tex, dV, q, tau);

        return molsim::detail::to_pyarray(std::move(tau));
    },
    "calc_tau(aij, gup, eup, frequencies, columndensity, Tex, dV, q)\n"
    "Array of optical depths at peak maximum for a series of transitions.\n"
    "\n"   
    "Parameters\n"
    "----------\n"
    "aij: NDArray[np.float64]\n"
    "   Array of Einstein A coefficients\n"
    "gup: NDArray[np.int32]\n"
    "   Array containing upper-state degeneracies\n"
    "frequencies: NDArray[np.float64]\n"
    "   Array containing transition frequencies in MHz\n"
    "columndensity: float\n"
    "   Column density of molecule, in cm^2\n"
    "Tex: float\n"
    "   Excitation temperature, in Kelvin\n"
    // This should be refactored throughout molsim to be dimensionless
    "dV: float\n"
    "   FWHM in velocity space (km/s)\n",
    "q: float\n"
    "   Partition function at T=Tex\n"
    "\n"
    "Returns\n"
    "-------\n"
    "NDArray[np.float64]\n"
    "   Array of optical depths at peak maximum.\n",
    "aij"_a, "gup"_a, "eup"_a, "frequencies"_a, "columndensity"_a, "Tex"_a, "dV"_a, "q"_a
    );

    functional.def("make_gaussians",
    [](nparray<double> x,
       nparray<double> centers,
       nparray<double> int0s,
       nparray<ssize_t> lls,
       nparray<ssize_t> uls,
       double dV) -> py::array_t<double>
    {
        AlignedVector<double> y;
        molsim::functional::make_gaussians(AlignedVector<double>(centers),
                                           AlignedVector<double>(int0s),
                                           AlignedVector<ssize_t>(lls),
                                           AlignedVector<ssize_t>(uls),
                                           dV,
                                           AlignedVector<double>(x),
                                           y);
        return molsim::detail::to_pyarray(std::move(y));
    },
    "make_gaussians(x, centers, int0s, lls, uls, dV)\n"
    "Sum of Gaussians parameterized by `centers` and `int0s`, calculated at `x`.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "x: NDArray[np.int64]\n"
    "   Array of points for which each gaussian will be calculated (MHz)\n"
    "centers: NDArray[np.float64]\n"
    "   Array of peak centers (MHz)\n"
    // This, along with calc_tau, should be refactored to take total integrated optical depth instead of peak maxima
    "int0s: NDArray[np.float64]\n",
    "   Array of peak maxima\n"
    "lls, uls: NDArray[np.int64]\n"
    "   Indices of `x` representing lower and upper limits for the nth gaussian,\n"
    "   beyond which the contribution is assumed to be zero.\n",
    "dV: float\n"
    "   FWHM in velocity space (km/s)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "NDArray[np.float64]\n"
    "   Array containing the sum of gaussians on the grid defined by `x`.\n",
    "x"_a, "centers"_a, "int0s"_a, "lls"_a, "uls"_a, "dV"_a);

    functional.def("calc_Ibg",
    [](nparray<double> freq,
       std::variant<double, nparray<double>> Tbg) -> py::array_t<double>
    {
        AlignedVector<double> ibg;
        if (std::holds_alternative<double>(Tbg))
            molsim::functional::calc_Ibg(AlignedVector<double>(freq),
                                         std::get<double>(Tbg),
                                         ibg);
        else
            molsim::functional::calc_Ibg(AlignedVector<double>(freq),
                                         AlignedVector<double>(std::get<nparray<double>>(Tbg)),
                                         ibg);
        return molsim::detail::to_pyarray(std::move(ibg));
    },
    "calc_Ibg(frequencies, Tbg)\n"
    "Calculates background flux densities (Jy/sr) given an array of frequencies (MHz) and thermal background temperature(s) (K).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "frequencies: NDArray[np.float64]\n"
    "   Frequencies (in MHz) at which to calculate background flux\n"
    "Tbg: float | NDArray[np.float64]\n"
    "   Either a single background temperature (K), or an array of frequency-dependent temperatures (K) for each element in `frequencies`\n"
    "\n"
    "Returns\n"
    "-------\n"
    "NDArray[np.float64]\n"
    "   Background flux density (Jy/sr) at each frequency.\n",
    "frequencies"_a, "Tbg"_a);

    functional.def("calc_Tb",
    [](nparray<double> frequency,
       nparray<double> tau,
       std::variant<double, nparray<double>> Tbg,
       double Tex) -> py::array_t<double>
    {
        AlignedVector<double> Tb;
        if (std::holds_alternative<double>(Tbg))
            molsim::functional::calc_Tb(AlignedVector<double>(frequency),
                                        AlignedVector<double>(tau),
                                        std::get<double>(Tbg),
                                        Tex,
                                        Tb);
        else
            molsim::functional::calc_Tb(AlignedVector<double>(frequency),
                                        AlignedVector<double>(tau),
                                        AlignedVector<double>(std::get<nparray<double>>(Tbg)),
                                        Tex,
                                        Tb);

        return molsim::detail::to_pyarray(std::move(Tb));
    },
    "Docstr",
    "frequency"_a, "tau"_a, "Tbg"_a, "Tex"_a);

    functional.def("apply_beam",
    [](nparray<double> frequencies,
       nparray<double> intensities,
       double source_size,
       double dish_size) -> py::tuple
    {
        auto [result, beam_dilution] = molsim::functional::apply_beam(AlignedVector<double>(frequencies),
                                                                      AlignedVector<double>(intensities),
                                                                      source_size,
                                                                      dish_size);
        return py::make_tuple(molsim::detail::to_pyarray(std::move(result)), molsim::detail::to_pyarray(std::move(beam_dilution)));
    },
    "apply_beam(frequencies, intensities, source_size, dish_size)\n"
    "",
    "frequencies"_a, "intensities"_a, "source_size"_a, "dish_size"_a
    );

    // The underlying code will be refactored here to not do unnecessary work,
    // but let's provide this functionality.
    functional.def("beam_dilution_factor",
    [](nparray<double> frequencies,
       double source_size,
       double dish_size) -> py::array_t<double>
    {
        AlignedVector<double> result;
        AlignedVector<double> f(frequencies);
        std::tie(result, std::ignore) = molsim::functional::apply_beam(f, f, source_size, dish_size);
        return molsim::detail::to_pyarray(std::move(result));
    },
    "Docstr",
    "frequencies"_a, "source_size"_a, "dish_size"_a);

    // Not worth realigning input data, as axpby doesn't assume aligned data
    functional.def("apply_vlsr",
    [](nparray<double> unshifted_frequencies,
       double vlsr) -> py::array_t<double>
    {
        const double scale = 1.0 - vlsr/ckm;
        const auto size = unshifted_frequencies.size();
        AlignedVector<double> result(size);
        axpby(size, scale, unshifted_frequencies.data(), 1, 0.0, result.data(), 1);
        return molsim::detail::to_pyarray(std::move(result));
    },
    "Docstr",
    "unshifted_frequencies"_a, "vlsr"_a);

    functional.def("compute_log_likelihood",
    [](const py::array_t<double, py::array::c_style | py::array::forcecast>& simulation,
       const py::array_t<double, py::array::c_style | py::array::forcecast>& obs_Tb,
       const py::array_t<double, py::array::c_style | py::array::forcecast>& obs_noise)
       -> std::variant<double,py::array_t<double>>
    {
        return molsim::functional::compute_log_likelihood(simulation, obs_Tb, obs_noise);
    },
    "Computes negative log-likelihood(s) given a simulation (or array of simulations) and the observed data.", "simulation"_a, "obs_Tb"_a, "obs_noise"_a);
}
