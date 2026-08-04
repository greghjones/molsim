#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "classes.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

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
}
