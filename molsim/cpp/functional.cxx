#include "util.hpp"
#include "constants.hpp"
#include "functional.hpp"

namespace molsim
{
namespace functional
{
void apply_vlsr(const AlignedVector<double>& infreq, const double vlsr, AlignedVector<double>& out)
{
    const double scale = 1.0 - vlsr/ckm;
    axpby(infreq.size(), scale, infreq.data(), 1, 0.0, out.data(), 1);
}
}
}
