#ifndef FORALLGENERATOR_HPP
#define FORALLGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/derivedparams.hpp>

namespace tinygemm{
namespace forallgen{

KernelString get_beta_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copya_kernelstring(const tinygemm::hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copyb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);
}
}







#endif
