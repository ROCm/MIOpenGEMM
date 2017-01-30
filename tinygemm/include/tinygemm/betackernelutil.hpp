#ifndef BETACKERNELUTIL_HPP
#define BETACKERNELUTIL_HPP



#include <stdlib.h>

namespace tinygemm{
namespace betac{
  
void set_betackernel_sizes(char fchar, bool isColMajor, bool tC, unsigned m, unsigned n, unsigned & dim_coal, unsigned & dim_uncoal, size_t & betac_global_work_size, size_t & betac_local_work_size);

std::string get_betac_kernel_string(char fchar, const std::string & kernelname);

}
}  // namespaces





#endif
