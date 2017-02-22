#include <tinygemm/tinygemmkernel.hpp>

#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemmerror.hpp>


#include <chrono>
namespace tinygemm{
  

TinyGemmKernel::TinyGemmKernel(cl_command_queue command_queue_, const std::string & hash_): command_queue(command_queue_), clprog(nullptr), clkern(nullptr), hash(hash_) {}
    
void TinyGemmKernel::try_release(){

  
  if (clprog != nullptr){
    openclutil::cl_release_program(clprog, "TinyGemmKernel Destructor");
  }
  if (clkern != nullptr){
    openclutil::cl_release_kernel(clkern, "TinyGemmKernel Destructor");
  }
}


void TinyGemmKernel::update(const KernelString & ks, outputwriting::OutputWriter & mowri){


  try_release();
  


  tgk_strings = ks;
  
  mowri << "compiling " << ks.type.basic << " ( " << ks.type.full << " ) ... " << Flush;


  auto start = std::chrono::high_resolution_clock::now();


  openclutil::set_program_and_kernel(command_queue, tgk_strings.kernstr, tgk_strings.fname, clprog, clkern);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms = end - start;
  float elapsed_seconds = fp_ms.count();


  mowri << "done in " << elapsed_seconds << " [s]" << Endl;

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


int TinyGemmKernel::enqueue(cl_uint num_events_in_wait_list, const cl_event *event_wait_list){
  cl_int ret;
  

  tgk_strings.global_work_size += 0;
  
  ret = clEnqueueNDRangeKernel(command_queue, clkern, 1, NULL, &tgk_strings.global_work_size, &tgk_strings.local_work_size, num_events_in_wait_list, event_wait_list, &clevent);
    
  if (ret != CL_OUT_OF_RESOURCES){
    openclutil::confirm_cl_status(ret, "in enqueue of TinyGemmKernel " + hash);
  }
  /* Either returning CL_SUCCESS or CL_OUT_OF_RESOURCES, any other bad result results in a throwd */
  return ret;
}

int TinyGemmKernel::enqueue(){
  return enqueue(0, nullptr);
}


void TinyGemmKernel::update_times(){
  //TODO : wrap clGetEventProfilingInfo in safety layer
  clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start, nullptr);
  clGetEventProfilingInfo(clevent, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr);
  v_times.push_back(1e-6*(t_end-t_start));    
}

void TinyGemmKernel::reset_times(){
  t_start = 0;
  t_end = 0;
  v_times.resize(0);
}


}
