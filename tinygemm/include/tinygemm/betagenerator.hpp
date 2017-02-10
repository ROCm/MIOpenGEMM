#ifndef BETAGENERATOR_HPP
#define BETAGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/derivedparams.hpp>

namespace tinygemm{
namespace betagen{

KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copya_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

}
}







#endif
