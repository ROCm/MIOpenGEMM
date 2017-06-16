#include <tinygemm/kernelstring.hpp>
#include <tinygemm/geometry.hpp>
#include <tinygemm/error.hpp>
#include <tinygemm/derivedparams.hpp>

namespace tinygemm{
namespace alphagen{
  
KernelString get_alpha_kernelstring(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const derivedparams::DerivedParams & dp);
 
}
}
