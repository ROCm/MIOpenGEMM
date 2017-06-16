#ifndef NORMALFORMGENERATOR_HPP
#define NORMALFORMGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/geometry.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/prepgenerator.hpp>

namespace tinygemm{
namespace nformgen{

KernelString get_nforma_kernelstring(const hyperparams::HyperParams & hp,  const TinyGemmGeometry & gg, const derivedparams::DerivedParams & dp);

KernelString get_nformb_kernelstring(const hyperparams::HyperParams & hp, const TinyGemmGeometry & gg, const derivedparams::DerivedParams & dp);
}
}







#endif
