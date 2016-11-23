#ifndef BETACKERNELUTIL_HPP
#define BETACKERNELUTIL_HPP



#include <stdlib.h>

namespace tinygemm{
namespace betac{
  
void set_betackernel_sizes(char fchar, bool isColMajor, bool tC, unsigned m, unsigned n, unsigned & dim_coal, unsigned & dim_uncoal, size_t & betac_global_work_size, size_t & betac_local_work_size);

extern const std::string cl_file_f32_path;
extern const std::string cl_file_f64_path;

std::string get_cl_file_path(char fchar);
}
}  // namespaces





#endif
