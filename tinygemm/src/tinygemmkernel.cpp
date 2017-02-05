#include <tinygemm/tinygemmkernel.hpp>

#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{
  
TinyGemmKernelStrings::TinyGemmKernelStrings(): type(""), kernstr(""), fname("") {}

TinyGemmKernel::TinyGemmKernel(cl_command_queue command_queue_, const std::string & hash_): command_queue(command_queue_), clprog(nullptr), clkern(nullptr), hash(hash_) {}
    
void TinyGemmKernel::try_release(){
  if (clprog != nullptr){
    openclutil::cl_release_program(clprog, "TinyGemmKernel Destructor");
  }
  if (clkern != nullptr){
    openclutil::cl_release_kernel(clkern, "TinyGemmKernel Destructor");
  }
}

//TODO : why is second string passed by const &, and not as rval?
void TinyGemmKernel::update(std::string && new_kernstr, const std::string & kern_func_name){
  try_release();
  tgk_strings.kernstr = new_kernstr;      
  tgk_strings.fname = kern_func_name;
  openclutil::set_program_and_kernel(command_queue, tgk_strings.kernstr, kern_func_name, clprog, clkern);
}

TinyGemmKernel::~TinyGemmKernel(){
  try_release();
}

bool TinyGemmKernel::is_set(){
  return (clprog != nullptr && clkern != nullptr);
}


void TinyGemmKernel::set_kernel_arg(cl_uint arg_index, size_t arg_size, const void *arg_value){
  
  if (clkern == nullptr){
    throw tinygemm_error("Attempt to set kernel argument of uninitialised kernel");
  }
  openclutil::cl_set_kernel_arg(clkern, arg_index, arg_size, arg_value, "in set_kernel_arg of TinyGemmKernel, " + hash + " index : " + std::to_string(arg_index));
}

void TinyGemmKernel::set_kernel_args(std::vector<std::pair<size_t, const void *> > arg_sizes_values){
  for (cl_uint arg_index = 0; arg_index < arg_sizes_values.size(); ++arg_index){
    set_kernel_arg(arg_index, arg_sizes_values[arg_index].first, arg_sizes_values[arg_index].second); 
  }
}

}
