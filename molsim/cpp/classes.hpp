#include <Python.h>
#include <string>
#include <optional>

#include "util.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Catalog
{
    public:
        Catalog(const py::object& catalog);
        uint64_t catid;
        std::string molecule;
        AlignedVector<double> frequency;
        AlignedVector<double> freq_err;
        AlignedVector<double> logint;
        AlignedVector<double> sijmu;
        AlignedVector<double> sij;
        AlignedVector<double> aij;
        AlignedVector<double> elow;
        AlignedVector<double> eup;
        AlignedVector<int> glow;
        AlignedVector<int> gup;
        bool measured;
};

class Molecule;

class PartitionFunction
{
    public:
        PartitionFunction(const py::object& qpart, Molecule* mol);
        PartitionFunction(const py::object& qpart) : PartitionFunction(qpart, nullptr) { };

        inline double q(double Tex) { return qrot(Tex)*qvib(Tex); };
        double qrot(double Tex);

        // not yet implemented, guarded in constructor
        inline double qvib(double) { return 1.0; };

    private:
        enum part_method { interpolation, counting } flag;
        double sigma;
        Molecule* parent_mol;
        AlignedVector<double> temps;
        AlignedVector<double> vals;

        double qrot_counting(double Tex);
};

class Molecule
{
    friend class PartitionFunction;

    public:
        Molecule(const py::object& mol_py);
        PartitionFunction qpart;
        Catalog catalog;

        inline double q   (double Tex) { return qpart.q(Tex); };
        inline double qrot(double Tex) { return qpart.qrot(Tex); };
        inline double qvib(double Tex) { return qpart.qvib(Tex); };
    protected:
        AlignedVector<double> level_degeneracies;
        AlignedVector<double> level_energies;
};

class Spectrum
{
    public:
        Spectrum();
        Spectrum(const py::object& spectrum_py);
        AlignedVector<double> freq0;  //unshifted frequency data
        AlignedVector<double> frequency; //frequency data
        AlignedVector<double> Tb; //intensity in units of [K]
        AlignedVector<double> Iv; //intensity in units of [Jy/beam] or [Jy/sr]
        AlignedVector<double> Tbg; //intensity of background in [K]
        AlignedVector<double> Ibg; //intensity of background in [Jy/beam] or [Jy/sr]
        AlignedVector<double> tau; //optical depths
        AlignedVector<double> tau_profile; //tau with line profile applied
        AlignedVector<double> freq_profile; //frequency of line profile data
        AlignedVector<double> int_profile; //intensity of line profile data
        AlignedVector<double> Tbg_profile; //background with line profile
        // velocity = None, //velocity space Data
        // int_sim = None, //intensity of a simulation
        // freq_sim = None, //frequency of a simulation
        // snr = None, //data in snr space
        // noise = None, //noise level with the same unit as intensity
        // id = None, //a unique ID for this spectrum
        // notes = None, //notes
        // name = None, //name
};

class Continuum
{
    public:
        Continuum();
        Continuum(const py::object& continuum);
        std::string cont_file;
        enum cont_type { thermal, interpolation, range } type;
        double params;
        AlignedVector<double> freqs;
        AlignedVector<double> temps;
        AlignedVector<double> fluxes;
        std::string notes;

        AlignedVector<double> Tbg(const AlignedVector<double>& freq) { AlignedVector<double> tbg(freq.size()); Tbg(freq, tbg); return tbg; };
        AlignedVector<double> Ibg(const AlignedVector<double>& freq)
        {
            AlignedVector<double> ibg(freq.size());
            if (type == thermal)
                Ibg(freq, params, ibg);
            else
                Ibg(freq, Tbg(freq), ibg);
            return ibg;
        };

        void Tbg(const AlignedVector<double>& freq, AlignedVector<double>& Tbg);
        void Ibg(const AlignedVector<double>& freq, double Tbg, AlignedVector<double>& Ibg);
        void Ibg(const AlignedVector<double>& freq, const AlignedVector<double>& Tbg, AlignedVector<double>& Ibg);
};

class Source
{
    public:
        Source();
        Source(const py::object& source);
        std::string name;
        double velocity;
        double size;
        std::optional<double> solid_angle;
        Continuum continuum;
        double column;
        double Tex; // this might be a vector later, will need to think about design here
        std::optional<double> Tkin;
        double dV;
        uint64_t id;
        std::string notes;
};

class Observatory
{
    public:
        Observatory();
        Observatory(const py::object& obs);
        std::string name;
        uint64_t id;
        bool sd;
        bool array;
        double dish;
        std::pair<double, double> synth_beam;
        // loc;
        AlignedVector<double> eta;
        enum eta_type_t
        {
            constant
        } eta_type;
        AlignedVector<double> eta_params;
        AlignedVector<double> atmo;
};

class Observation
{
    public:
        std::string name;
        // py::object coords;
        double vlsr;
        Spectrum spectrum;
        std::optional<Observatory> observatory;
        uint64_t id;
        std::string notes;

        Observation();
        Observation(const py::object& obs);
};

class Simulation
{
    public:
        Simulation(const py::object& spectrum = py::none(),
                   const py::object& observation = py::none(),
                   const py::object& source = py::none(),
                   const py::array_t<double>& ll = py::array_t<double>(),
                   const py::array_t<double>& ul = py::array_t<double>(),
                   const std::string& line_profile = "gaussian",
                   double sim_width = 10.0,
                   double res = 0.010,
                   const py::object& mol = py::none(),
                   std::string units = "K",
                   const std::string& notes = "",
                   bool use_obs = false,
                   bool add_noise_flag = false,
                   double noise = 0.0,
                   double tau_threshold = 0.0,
                   double eup_threshold = 0.0);

        Spectrum spectrum;
        std::optional<Observation> observation;
        Source source;
        AlignedVector<double> ll;
        AlignedVector<double> ul;
        enum line_profile_type
        {
            NoLineProfile,
            Gaussian
        } line_profile;
        double sim_width;
        double res;
        Molecule mol;
        enum unit_type
        {
            K, mK, Jy_beam
        } units;
        std::string notes;
        bool use_obs;
        bool add_noise_flag;
        double noise;
        double tau_threshold;
        double eup_threshold;

        AlignedVector<double> aij;
        AlignedVector<int> gup;
        AlignedVector<double> eup;
        AlignedVector<double> beam_dilution;

        Source get_source() { return source; }
        void set_source(const py::object& s) { source = Source(s); }

        void set_line_profile(std::string label);
        void set_units();
        void update();
        void apply_voffset();
        void calc_tau();
        void make_lines();
        void beam_correct();

    private:
        void set_arrays();
        void calc_bg();
        void calc_Iv() { };
        void calc_Tb(const AlignedVector<double> frequency,
                     const AlignedVector<double>& tau,
                     const AlignedVector<double>& Tbg,
                     double Tex,
                           AlignedVector<double>& Tb);
        void calc_Tb(const AlignedVector<double> frequency,
                     const AlignedVector<double>& tau,
                     const double Tbg,
                     const double Tex,
                           AlignedVector<double>& Tb);
        void apply_eta();
        void add_noise() { };
        void make_gaussians(const AlignedVector<double>& centers, const AlignedVector<double>& int0s,
                            const AlignedVector<long>& lls, const AlignedVector<long>& uls, double dV,
                            const AlignedVector<double>& x, AlignedVector<double>& y);

        AlignedVector<long> l_idxs;
        AlignedVector<long> u_idxs;
        bool doonce = true;
};

