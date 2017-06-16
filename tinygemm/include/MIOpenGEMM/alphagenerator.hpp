#include <MIOpenGEMM/kernelstring.hpp>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/derivedparams.hpp>

namespace MIOpenGEMM{
namespace alphagen{
  
KernelString get_alpha_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);
 
}
}
