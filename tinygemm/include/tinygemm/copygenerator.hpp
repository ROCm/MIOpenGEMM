#ifndef COPYGENERATOR_HPP
#define COPYGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>

namespace tinygemm{
namespace copygen{

KernelString get_copy_a_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp );

}
}







#endif
