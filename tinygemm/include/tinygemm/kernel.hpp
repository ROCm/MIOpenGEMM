#ifndef TINYGEMM_TINYGEMMKERNEL_HPP
#define TINYGEMM_TINYGEMMKERNEL_HPP

#include <vector> 
#include <algorithm>

#include <CL/cl.h>
#include <tinygemm/kernelstring.hpp>
#include <tinygemm/outputwriter.hpp>

namespace MIOpenGEMM{
  

class TinyGemmKernel{
  
  public:
    cl_command_queue command_queue;    
    KernelString tgk_strings;
  
    /* used for getting performance of kernel */
    cl_event clevent;

    /* stores (the most recent of n_runs) execution time */  
    size_t t_start;
    size_t t_end;
    std::vector<float> v_times;
      
  private:
    cl_program clprog;

  public:
    cl_kernel clkern;
  
  public:
    std::string hash;


  private:
    void try_release();
    void set_kernel_arg(cl_uint arg_index, size_t arg_size, const void *arg_value);
  
  public:  
    TinyGemmKernel(cl_command_queue command_queue_, const std::string & hash_);
    
    TinyGemmKernel():TinyGemmKernel(nullptr, "default constructed TinyGemmKernel"){}
    
    void update(const KernelString & ks, outputwriting::OutputWriter & mowri); 

  
    ~TinyGemmKernel();
    
    TinyGemmKernel & operator= (const TinyGemmKernel &) = default;

    bool is_set();
    void set_kernel_args(std::vector<std::pair<size_t, const void *> > arg_sizes_values);
    
    int enqueue(cl_uint num_events_in_wait_list, const cl_event *event_wait_list);
    int enqueue();
    
    void update_times();
    
    void reset_times();
};

}

#endif
