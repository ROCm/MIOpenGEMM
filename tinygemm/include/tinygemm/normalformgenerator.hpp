#ifndef NORMALFORMGENERATOR_HPP
#define NORMALFORMGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/derivedparams.hpp>

namespace tinygemm{
namespace forallgen{

KernelString get_nforma_kernelstring(const tinygemm::hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_nformb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);
}
}







#endif
