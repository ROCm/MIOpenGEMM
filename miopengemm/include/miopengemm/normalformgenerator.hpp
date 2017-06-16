#ifndef NORMALFORMGENERATOR_HPP
#define NORMALFORMGENERATOR_HPP

#include <miopengemm/kernelstring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/prepgenerator.hpp>

namespace MOOMOOMOOGEMM{
namespace nformgen{

KernelString get_nforma_kernelstring(const hyperparams::HyperParams & hp,  const Geometry & gg, const derivedparams::DerivedParams & dp);

KernelString get_nformb_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);
}
}







#endif
