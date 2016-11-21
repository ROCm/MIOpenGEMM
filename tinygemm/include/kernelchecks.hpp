#ifndef KERNELCHECKS_HPP
#define KERNELCHECKS_HPP

#include <algorithm>
#include <map>

namespace kernelutil{
  
void run_preprocessor_parameter_tests(std::map<std::string, unsigned> ipps, std::map<std::string, std::string> spps,

bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);  

void check_gpu_kernel_filename(std::string kernel_filename);

void check_gpu_kernel_filenames(const std::vector<std::vector<std::string>> & gpu_kernel_filenames);

void check_gpu_kernels_preprocessor_parameters(const std::vector<std::vector<std::string>> & gpu_kernel_filenames, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);

void check_gpu_kernel_preprocessor_parameters(std::string gpu_kernel_filename, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring);


void check_gpu_kernel_filename(std::string kernel_filename);


}

#endif
