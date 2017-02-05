#ifndef TINYGEMM_TINYGEMMKERNEL_HPP
#define TINYGEMM_TINYGEMMKERNEL_HPP

#include <vector> 
#include <algorithm>

#include <CL/cl.h>

namespace tinygemm{
  
class TinyGemmKernelStrings{
public:
  /*betac_alphab, betac_workspace, etc.*/
  std::string type;  
  std::string kernstr;
  std::string fname;
  TinyGemmKernelStrings();

};

class TinyGemmKernel{
  
  public:
    cl_command_queue command_queue;    
    TinyGemmKernelStrings tgk_strings;
  
  private:
    cl_program clprog;

  public:
    cl_kernel clkern;
  
  private:
    std::string hash;


  private:
    void try_release();
    void set_kernel_arg(cl_uint arg_index, size_t arg_size, const void *arg_value);
  
  public:  
    TinyGemmKernel(cl_command_queue command_queue_, const std::string & hash_);
    //TODO : why is second string passed by const &, and not as rval?
    void update(std::string && new_kernstr, const std::string & kern_func_name);
    ~TinyGemmKernel();
    bool is_set();
    void set_kernel_args(std::vector<std::pair<size_t, const void *> > arg_sizes_values);
};

}

#endif
