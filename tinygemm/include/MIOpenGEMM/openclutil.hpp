#ifndef OPENCLUTIL_H
#define OPENCLUTIL_H


#include <MIOpenGEMM/outputwriter.hpp>
#include <CL/cl.h>


namespace MIOpenGEMM{
namespace openclutil{




/* hash can be any string to help locate the problem (exception) should one arise */
void confirm_cl_status(cl_int ret, const std::string & hash = "", const std::string & function = "unknown");


void set_platform_etc(cl_platform_id & platform, cl_uint & num_platforms, cl_context & context, cl_device_id & device_id_to_use, outputwriting::OutputWriter & mowri);

void set_program_and_kernel(const cl_command_queue & command_queue, const std::string & kernel_string, const std::string & kernel_function_name, cl_program & program, cl_kernel & kernel, outputwriting::OutputWriter & mowri);
  
void cl_release_kernel(cl_kernel kernel, const std::string & hash);

void cl_release_context(cl_context context, const std::string & hash);  

cl_command_queue cl_create_command_queue(cl_context context, cl_device_id device, cl_command_queue_properties properties, const std::string & hash);

void cl_release_command_queue(cl_command_queue command_queue, const std::string & hash);
    
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

void cl_get_event_profiling_info(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);
  
void get_device_info_from_command_queue(cl_command_queue command_queue, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);

void get_platform_info_from_command_queue(cl_command_queue command_queue, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);

cl_program cl_create_program_with_source(cl_context context, cl_uint count, const char **strings, const size_t *lengths, const std::string & hash);

cl_kernel cl_create_kernel(cl_program program, const char *kernel_name, const std::string & hash);

void cl_get_program_info(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash);

cl_mem cl_create_buffer_from_command_queue(cl_command_queue command_queue, cl_mem_flags flags, size_t size, void * host_ptr, const std::string & hash);    

cl_command_queue auto_get_command_queue(outputwriting::OutputWriter & mowri,  cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE);

void cl_enqueue_write_buffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event, const std::string & hash);

void cl_enqueue_read_buffer(cl_command_queue command_queue,cl_mem buffer,cl_bool blocking_read,size_t offset,size_t cb,void *ptr,cl_uint num_events_in_wait_list,
const cl_event *event_wait_list,cl_event *event, const std::string & hash);


void cl_build_program(cl_program program,cl_uint num_devices,const cl_device_id *device_list,const char *options,void (*pfn_notify)(cl_program, void *user_data),void *user_data, outputwriting::OutputWriter & mowri, const std::string & hash);

class SafeClMem{
  public:
    cl_mem clmem;
    std::string hash;
    SafeClMem(const std::string & hash); 
    
    ~SafeClMem();
};

class CommandQueueInContext{
  public:
    cl_command_queue command_queue;
    std::string hash;  
    CommandQueueInContext(outputwriting::OutputWriter & mowri, const std::string & hash);
    ~CommandQueueInContext();
    
};

  
class OpenCLPlatformInfo{
  public:
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;    

    OpenCLPlatformInfo(cl_platform_id platform_id);
    OpenCLPlatformInfo() = default;
    std::string get_string() const;

};



class OpenCLDeviceInfo{
  public:


    OpenCLPlatformInfo platinfo;

    std::string device_name;  
    std::string device_version;  
    std::string driver_version;
    std::string identifier;
    bool device_available;
    size_t device_global_mem_size;
    unsigned device_max_clock_frequency;
    unsigned device_max_compute_units;
    unsigned device_max_work_group_size;
    unsigned wg_atom_size;

    std::string get_string() const;
    OpenCLDeviceInfo(const cl_command_queue & command_queue);
    OpenCLDeviceInfo();
};


} // end namesapce
}

#endif
