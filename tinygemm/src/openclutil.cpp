#include <string>
#include <sstream>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/kernelsnips.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/openclutil.hpp>
#include  <CL/cl.h>


namespace tinygemm{
namespace openclutil{







void confirm_cl_status(cl_int ret, const std::string & hash, const std::string & function){
  if (ret != CL_SUCCESS){
    std::stringstream errms;
    errms << "Reporting an opencl error (grep this code: `" << hash << "') which returned with cl_int " << ret << " from function " << function << "." << std::endl;
    throw tinygemm_error(errms.str());
  }
}





  void cl_release_kernel(cl_kernel kernel, const std::string & hash){
    cl_int ret = clReleaseKernel(kernel);
    confirm_cl_status(ret, hash, "cl_release_kernel");
  }
  
  void cl_release_program(cl_program program, const std::string & hash){
    cl_int ret = clReleaseProgram(program);
    confirm_cl_status(ret, hash, "cl_release_program");
  }
 
 
  void cl_set_kernel_arg(cl_kernel & kernel, cl_uint arg_index, size_t arg_size, const void * arg_value, const std::string & hash){
    cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    confirm_cl_status(ret, hash, "cl_set_kernel_arg");
  }


  void cl_flush(cl_command_queue command_queue, const std::string & hash){
    cl_int ret = clFlush(command_queue);
    confirm_cl_status(ret, hash, "cl_flush"); 
  }
  
  void cl_wait_for_events(cl_uint num_events, const cl_event * event_list, const std::string & hash){
    cl_int ret = clWaitForEvents(num_events, event_list);
    confirm_cl_status(ret, hash, "cl_wait_for_events");
  }


  void cl_get_command_queue_info(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret, const std::string & hash){
    cl_int ret = clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
    confirm_cl_status(ret, hash, "cl_get_command_queue_info");
  }


  cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void * host_ptr, const std::string & hash){
    cl_int errcode_ret;
    cl_mem toreturn = clCreateBuffer(context, flags, size, host_ptr, &errcode_ret);
    confirm_cl_status(errcode_ret, hash, "cl_create_buffer");
    return toreturn;
  }
   
  void cl_enqueue_copy_buffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, const std::string & hash){
    cl_int ret = clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, cb, num_events_in_wait_list, event_wait_list, event); 
    confirm_cl_status(ret, hash, "cl_enqueue_copy_buffer");
  }
 

  void cl_release_mem_object(cl_mem memobj, const std::string & hash){
    cl_int ret = clReleaseMemObject(memobj);
    confirm_cl_status(ret, hash, "cl_release_mem_object");
  }   
  
  void cl_enqueue_ndrange_kernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t * global_work_offset, const size_t * global_work_size, const size_t * local_work_size,cl_uint num_events_in_wait_list, const cl_event *event_wait_list,cl_event * event, const std::string & hash){
    cl_int ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
   confirm_cl_status(ret, hash, "cl_enqueue_ndrange_kernel");
  }
  

  cl_mem cl_create_buffer_from_command_queue(cl_command_queue command_queue, cl_mem_flags flags, size_t size, void * host_ptr, const std::string & hash){
    cl_context context;
    cl_get_command_queue_info(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr, hash + " + (cl_create_buffer_from_command_queue)");
    return cl_create_buffer(context, flags, size, host_ptr, hash + "+ (cl_create_buffer_from_command_queue)");
  }

  void cl_get_platform_ids(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms, const std::string & hash){
    cl_int ret = clGetPlatformIDs(num_entries, platforms, num_platforms);
    confirm_cl_status(ret, hash, "cl_get_platform_ids");
  }
  

  cl_context cl_create_context_from_type(cl_context_properties * properties, cl_device_type  device_type, void  (*pfn_notify) (const char *errinfo, const void  *private_info, size_t  cb, void  *user_data), void  *user_data, const std::string & hash){
    cl_int errcode;
    cl_context context = clCreateContextFromType(properties, device_type, pfn_notify, user_data, & errcode);
    confirm_cl_status(errcode, hash, "cl_create_context_from_type");
    return context;
  }




  void  cl_get_context_info(cl_context context, cl_context_info param_name, size_t param_value_size, void * param_value, size_t * param_value_size_ret, const std::string & hash){
    cl_int ret = clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret);
    confirm_cl_status(ret, hash, "cl_get_context_info");
  }
  
  
  void cl_get_device_info(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash){
    cl_int ret = clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
    confirm_cl_status(ret, hash, "cl_get_device_info");
  }
    


  cl_kernel cl_create_kernel(cl_program program, const char *kernel_name, const std::string & hash){
    cl_int errcode_ret;
    cl_kernel kernel = clCreateKernel(program, kernel_name, & errcode_ret);
    
    /* proof that release should be called once, even if copies have been made */
    //std::cout << "in create kernel" << std::endl;
    //cl_kernel kernel2 = kernel;
    //cl_release_kernel(kernel2, "bla bla");
    //std::cout << "boom" << std::endl;
   
    //SafeClMem mem2("mem2");
    //cl_release_mem_object(mem2, "bla bla");
    
    
    confirm_cl_status(errcode_ret, hash, "cl_create_kernel");
    return kernel;
  }
  






cl_program cl_create_program_with_source(cl_context context, cl_uint count, const char **strings, const size_t *lengths, const std::string & hash){  
  cl_int errcode_ret;
  cl_program program = clCreateProgramWithSource(context, count, strings, lengths,  &errcode_ret);
  confirm_cl_status(errcode_ret, hash, "cl_create_program_with_source");
  return program;
}


void cl_build_program(cl_program program,cl_uint num_devices,const cl_device_id *device_list,const char *options,void (*pfn_notify)(cl_program, void *user_data),void *user_data, const std::string & hash){
  cl_int ret = clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
  
  
  if (ret != CL_SUCCESS){
    char buffer[10240];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
    fprintf(stderr, "CL Compilation failed:\n%s", buffer);
  }

  confirm_cl_status(ret, hash, "cl_build_program");
  
}



void cl_get_program_info(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret, const std::string & hash){
  cl_int ret = clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
  confirm_cl_status(ret, hash, "void cl_get_program_info");
}










/* TODO : get rid of these of these raw calls to cl_* */
void set_platform_etc(cl_platform_id & platform, cl_uint & num_platforms, cl_context & context, cl_device_id & device_id_to_use, outputwriting::OutputWriter & mowri){
  
  /* Get the platform(s) */
  cl_get_platform_ids(1, &platform, &num_platforms, "in set_platform_etc");
  //cl_int status;
  
  /* Create context */
  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
  cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
  context = cl_create_context_from_type(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, "in set_platform_etc...");
  //if (status != CL_SUCCESS){
    //std::string errm("There's been an error in set_platform_etc in openclutil.cpp: the return status of clCreateContextFromType is not CL_SUCCESS. The status returned is ");
    //errm += status;
    //errm + ". ";
    //throw tinygemm_error(errm);
  //}
  
  /* Query the number of devices */
  int deviceListSize;
  cl_get_context_info(context, CL_CONTEXT_NUM_DEVICES, sizeof(int), &deviceListSize, nullptr, "getting deviceListSize");
  
  if (deviceListSize == 0){
    throw tinygemm_error("There are no devices detected. \nSpecifically, using clGetContextInfo with CL_CONTEX_NUM_DEVICES as the flag returns 0. \nThis error is being thrown from set_platform_etc in openclutil.cpp. Please have a look...");
  }

  
  std::vector<cl_device_id> devices(deviceListSize);
  
  /* Get the devices */
  cl_get_context_info(context, CL_CONTEXT_DEVICES, deviceListSize*sizeof(cl_device_id), devices.data(), nullptr, "getting the devices");
  char deviceName[100];
  cl_uint max_compute_units;  
  
  /* Go through the devices and see how many compute units each has, storing the best along the way */
  cl_uint max_max_compute_units = 0;
  char bestDeviceName[100];
  cl_device_id bestDeviceId; 
  
  std::string device_compute_unit_count_string = "The following devices were detected: \n";

  for (int i = 0; i < deviceListSize; ++i){
    /* get name and number of compute units of the device whose id is devices[i] */
    

    cl_get_device_info(devices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr, "getting CL_DEVICE_NAME");
    cl_get_device_info(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, nullptr, "getting CL_DEVICE_MAX_COMPUTE_UNITS");
    
    if (max_compute_units > max_max_compute_units){
      max_max_compute_units = max_compute_units;
      cl_get_device_info(devices[i], CL_DEVICE_NAME, sizeof(bestDeviceName), bestDeviceName, nullptr, "getting best devices CL_DEVICE_NAME");
      bestDeviceId = devices[i];
    }

    device_compute_unit_count_string += deviceName;
    device_compute_unit_count_string += ", which as ";
    device_compute_unit_count_string += std::to_string(max_compute_units);
    device_compute_unit_count_string +=  " compute units\n";
  }
  
  
  bool only_good_hardware = true;
  if (only_good_hardware == true && max_max_compute_units < 40){
    std::string errm = device_compute_unit_count_string;
    errm += "As this is less than 64, an error is being thrown. \nIf you wish to use a device with fewer than 64 CUs, please make changes here (in openclutil.cpp)";
    throw tinygemm_error(errm);
  }
  
  
  else{
    mowri << "Will use device " << bestDeviceName << ", which has " << max_max_compute_units << " CUs. \nTo use a different device, consider modifying set_platform_etc in openclutil.cpp (or write custom OpenCL boilerplate)." << Endl;
    device_id_to_use = bestDeviceId;
  }

}



void set_program_and_kernel(const cl_command_queue & command_queue, cl_program & program, cl_kernel & kernel, std::string & kernel_function_name, const std::string & kernel_string){
  
  
  
  cl_context context;
  cl_get_command_queue_info(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL, "hashIDbla102");
  
  cl_device_id device_id_to_use;
  cl_get_command_queue_info(command_queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id_to_use, NULL, "hashIDbla101");
        
  kernel_function_name = kernelutil::get_kernel_function_name(kernel_string);
  
  
  
  auto kernel_cstr = kernel_string.c_str();
  auto kernel_string_size =  kernel_string.size();
  program = cl_create_program_with_source(context, 1, &kernel_cstr, &kernel_string_size , "creating program in set_program_and_kernel");
  
  /* To generate isa code, you'll need to add something like :   ``` -save-temps= + "/some/path/"  '''    to the following string  */
  std::string buildOptions_11 = "-cl-std=CL2.0";
  auto buildOptions = buildOptions_11.c_str();
  cl_build_program(program, 1, &device_id_to_use, buildOptions, NULL, NULL, "building program in set_program_and_kernel"); 
  
  
  
  /* writing binary : from http://www.cs.bris.ac.uk/home/simonm/montblanc/AdvancedOpenCL_full.pdf */
  if (false){
    /* TODOEVENTUALLY This will break if more than one device used */
    size_t size; 
    /* the size of the binary is ridiculous. 60 MB!! */
    cl_get_program_info(program, CL_PROGRAM_BINARY_SIZES, 0, &size, NULL,  "getting CL_PROGRAM_BINARY_SIZES ");
    std::vector<unsigned char> v_binary (size + 10);
    cl_get_program_info(program, CL_PROGRAM_BINARIES, size, v_binary.data(), NULL, "getting CL_PROGRAM_BINARIES "); 
    //std::string filename_bin(filename);
    std::string filename_bin = "bladibla_bin";
    FILE *filebinary = fopen(filename_bin.c_str(), "w");
    fwrite(v_binary.data(), 1, size, filebinary);
    /* the binary file has been written to filename_bin */ 
    throw tinygemm_error("go look at created binary file!");
    /* done */
  }
  
  
  kernel = cl_create_kernel(program, kernel_function_name.c_str(), "getting kernel in set_program_and_kernel");  
}
  



    
}
} // end namespace

