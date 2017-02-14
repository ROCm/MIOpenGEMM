#ifndef TENTOTIN_HPP
#define TENTOTIN_HPP

#include <tinygemm/tinygemmgeometry.hpp>


namespace tinygemm{
namespace tensilegen{
  
KernelString get_tensile_kernelstring(const tinygemm::TinyGemmGeometry & gg);

}
}
#endif
