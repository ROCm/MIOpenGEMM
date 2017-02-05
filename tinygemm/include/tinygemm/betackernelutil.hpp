#ifndef BETACKERNELUTIL_HPP
#define BETACKERNELUTIL_HPP



#include <stdlib.h>

#include <tinygemm/tinygemmgeometry.hpp>
namespace tinygemm{
namespace betac{

const size_t work_per_thread = 4;
const size_t n_work_items_per_group = 64;

size_t get_global_work_size(const tinygemm::TinyGemmGeometry & gg);

std::string get_betac_kernel_string(char fchar, const std::string & kernelname);

}
}  // namespaces





#endif
