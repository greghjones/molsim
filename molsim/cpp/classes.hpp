#include <string>
#include <optional>

#include "util.hpp"

class Catalog
{
    public:
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

class Molecule
{
    public:
        Catalog catalog;

        double q(double Tex);
};

class Spectrum
{
    public:
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
        std::string cont_file;
        enum type { thermal, interpolation, range };
        // params; may need adapting
        AlignedVector<double> freqs;
        AlignedVector<double> temps;
        AlignedVector<double> fluxes;
        std::string notes;
};

class Source
{
    public:
        std::string name;
        double velocity;
        double size;
        double solid_angle;
        Continuum continuum;
        double column;
        double Tex; // this might be a vector later, will need to think about design here
        double Tkin;
        double dV;
        uint64_t id;
        std::string notes;
};

class Observatory
{
    public:
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
        // coords;
        double vlsr;
        Spectrum spectrum;
        std::optional<Observatory> observatory;
        uint64_t id;
        std::string notes;
};

class Simulation
{
    public:
        Simulation(Spectrum spectrum = Spectrum(),
                   std::optional<Observation> observation = {},
                   Source source = Source(),
                   AlignedVector<double> ll = AlignedVector<double>(),
                   AlignedVector<double> ul = AlignedVector<double>(),
                   const std::string& line_profile = "gaussian",
                   double sim_width = 10.0,
                   double res = 10.0,
                   Molecule mol = Molecule(),
                   const std::string& units = "K",
                   const std::string& notes = "",
                   bool use_obs = false,
                   bool add_noise = false,
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
        bool add_noise;
        double noise;
        double tau_threshold;
        double eup_threshold;

        AlignedVector<double> aij;
        AlignedVector<int> gup;
        AlignedVector<double> eup;

        void set_line_profile(std::string label);
        void set_units(std::string label);
        void update();

    private:
        void set_arrays();
        void apply_voffset();
        void calc_tau();
        void calc_bg();
        void calc_Iv();
        void calc_Tb();
        void beam_correct();
        void apply_eta();
        void make_lines();
        void add_noise_();
        
};

