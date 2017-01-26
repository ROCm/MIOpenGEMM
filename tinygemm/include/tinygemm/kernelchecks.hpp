#ifndef KERNELCHECKS_HPP
#define KERNELCHECKS_HPP

#include <algorithm>
#include <map>

namespace tinygemm{
namespace kernelutil{
  
void run_preprocessor_parameter_tests(std::map<std::string, unsigned> ipps, std::map<std::string, std::string> spps, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);  


void check_gpu_kernels_preprocessor_parameters(const std::vector<std::vector<std::string>> & gpu_kernel_strings, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);

void check_gpu_kernel_preprocessor_parameters(const std::string & gpu_kernel_string, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);



}
}

#endif
