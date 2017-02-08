#ifndef BETAGENERATOR_HPP
#define BETAGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>


namespace tinygemm{
namespace betagen{

KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg);

}
}







#endif
