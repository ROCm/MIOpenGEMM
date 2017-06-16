#ifndef NORMALFORMGENERATOR_HPP
#define NORMALFORMGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/geometry.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/prepgenerator.hpp>

namespace tinygemm{
namespace nformgen{

KernelString get_nforma_kernelstring(const tinygemm::hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_nformb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);
}
}







#endif
