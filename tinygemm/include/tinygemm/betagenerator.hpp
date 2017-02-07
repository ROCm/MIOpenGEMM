#ifndef BETAGENERATOR_HPP
#define BETAGENERATOR_HPP

#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmgeometry.hpp>


namespace tinygemm{
namespace betagen{

KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg);

}
}




////size_t get_global_work_size(const tinygemm::TinyGemmGeometry & gg);

////interface to change, temp hack
////size_t get_local_work_size(const tinygemm::TinyGemmGeometry & gg);

////std::string get_betac_kernel_string(char fchar, const std::string & kernelname);

//}
//}  // namespaces





#endif
