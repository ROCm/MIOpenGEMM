#ifndef OPENCLUTIL_H
#define OPENCLUTIL_H


#include "outputwriter.hpp"
#include <CL/cl.h>

namespace tinygemm{
namespace openclutil{




/* hash can be any string to help locate the problem (exception) should one arise */
void confirm_cl_status(cl_int ret, const std::string & hash = "", const std::string & function = "unknown");

void set_platform_etc(cl_platform_id & platform, cl_uint & num_platforms, cl_context & context, cl_device_id & device_id_to_use, outputwriting::OutputWriter & mowri);

void set_program_and_kernel(const cl_command_queue & command_queue, cl_program & program, cl_kernel & kernel, std::string & kernel_function_name, const std::string & kernel_string);  
  
void cl_release_kernel(cl_kernel kernel, const std::string & hash);

void cl_release_program(cl_program program, const std::string & hash);

void cl_set_kernel_arg(cl_kernel & kernel, cl_uint arg_index, size_t arg_size, const void * arg_value, const std::string & hash);

void cl_flush(cl_command_queue command_queue, const std::string & hash);

void cl_wait_for_events(cl_uint num_events, const cl_event * event_list, const std::string & hash);

void cl_get_command_queue_info(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret, const std::string & hash);

cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void * host_ptr, const std::string & hash);

void cl_enqueue_copy_buffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, const std::string & hash);

void cl_release_mem_object(cl_mem memobj, const std::string & hash);

void cl_enqueue_ndrange_kernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t * global_work_offset, const size_t * global_work_size, const size_t * local_work_size,cl_uint num_events_in_wait_list, const cl_event *event_wait_list,cl_event * event, const std::string & hash);

void cl_get_platform_ids(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms);

cl_context cl_create_context_from_type(cl_context_properties * properties, cl_device_type  device_type, void  (*pfn_notify) (const char *errinfo, const void  *private_info, size_t  cb, void  *user_data), void  *user_data);

void  cl_get_context_info(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t * param_value_size_ret, const std::string & hash);  

void cl_get_device_info(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);

cl_program cl_create_program_with_source(cl_context context, cl_uint count, const char **strings, const size_t *lengths, const std::string & hash);

cl_kernel cl_create_kernel(cl_program program, const char *kernel_name, const std::string & hash);

void cl_get_program_info(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);

cl_mem cl_create_buffer_from_command_queue(cl_command_queue command_queue, cl_mem_flags flags, size_t size, void * host_ptr, const std::string & hash);    

/* TODO : move defn to cpp*/
class SafeClMem{
  public:
    cl_mem clmem;
    std::string hash;
    SafeClMem(const std::string & hash):clmem(nullptr),hash(hash) {};
    ~SafeClMem(){
      if (clmem != nullptr){
        openclutil::cl_release_mem_object(clmem, hash);
      }
    }
};

class SafeClProgAndKern{

    //safe_main_prog_and_kern.clkern = kernel;
    //safe_main_prog_and_kern.clprog = program;


public:
    cl_program clprog;
    cl_kernel clkern;
    std::string hash;

    SafeClProgAndKern(const std::string & hash): clprog(nullptr), clkern(nullptr), hash(hash) {};
    SafeClProgAndKern(cl_program clprog, cl_kernel clkern, const std::string & hash): clprog(clprog), clkern(clkern), hash(hash) {};
    ~SafeClProgAndKern(){
      if (clprog != nullptr){
        openclutil::cl_release_program(clprog, hash);
      }
      if (clkern != nullptr){
        openclutil::cl_release_kernel(clkern, hash);
      }
    }
};
  



} // end namesapce
}

#endif
