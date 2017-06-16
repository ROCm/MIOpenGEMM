#ifndef NORMALFORMGENERATOR_HPP
#define NORMALFORMGENERATOR_HPP

#include <MIOpenGEMM/kernelstring.hpp>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/derivedparams.hpp>
#include <MIOpenGEMM/prepgenerator.hpp>

namespace MIOpenGEMM{
namespace nformgen{

KernelString get_nforma_kernelstring(const hyperparams::HyperParams & hp,  const Geometry & gg, const derivedparams::DerivedParams & dp);

KernelString get_nformb_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp);
}
}







#endif
