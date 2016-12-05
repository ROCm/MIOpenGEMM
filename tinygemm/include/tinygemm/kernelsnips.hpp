#ifndef KERNELSNIPS_HPP
#define KERNELSNIPS_HPP

#include <algorithm>
#include <map>

namespace tinygemm{
namespace kernelutil{

std::pair<std::map<std::string, unsigned>, std::map<std::string, std::string>> 
get_all_preprocessor_parameters(std::string kernel_filename);

std::map<std::string, unsigned> 
get_integer_preprocessor_parameters(std::string kernel_filename);

std::string 
get_kernel_function_name(std::string kernel_filename);

void
set_sizes_from_kernel_source(unsigned & macro_tile_width, unsigned & macro_tile_height, unsigned & n_workitems_per_workgroup, unsigned & n_work_items_per_c_elm, unsigned & does_beta_c_inc, std::string kernelfilename);

std::string
get_as_single_string(std::string filename);

} //namespace
}

#endif
