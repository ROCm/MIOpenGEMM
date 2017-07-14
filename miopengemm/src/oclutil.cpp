/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <CL/cl.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace oclutil
{

cl_mem get_copy(cl_command_queue command_queue, cl_mem c, size_t c_nbytes, const std::string& hash)
{

  cl_mem   c_copied;
  cl_event c_copy_event;

  oclutil::cl_set_buffer_from_command_queue(c_copied,
                                            command_queue,
                                            CL_MEM_READ_WRITE,
                                            c_nbytes,
                                            NULL,
                                            hash + ", in function get_copy which returns a cl_mem",
                                            true);

  oclutil::cl_enqueue_copy_buffer(command_queue,
                                  c,
                                  c_copied,
                                  0,
                                  0,
                                  c_nbytes,
                                  0,
                                  NULL,
                                  &c_copy_event,
                                  hash + ", in function get_copy which returns a cl_mem",
                                  true);

  oclutil::cl_wait_for_events(1, &c_copy_event, "in function find", true);
  return c_copied;
}

const std::string fiji_string("gfx803");
const std::string vega_string("gfx900");

Result
confirm_cl_status(cl_int ret, const std::string& hash, const std::string& function, bool strict)
{
  std::stringstream errms;
  errms << "";

  if (ret != CL_SUCCESS)
  {
    errms << "Reporting an opencl error (grep this code: `" << hash
          << "') which returned with cl_int " << ret << " from function " << function << "."
          << std::endl;
  }

  if (strict == true && ret != CL_SUCCESS)
  {
    throw miog_error(errms.str());
  }

  else
  {
    return Result(ret, errms.str());
  }
}

Result cl_set_command_queue(cl_command_queue&           a_cl_command_queue,
                            cl_context                  context,
                            cl_device_id                device,
                            cl_command_queue_properties properties,
                            const std::string&          hash,
                            bool                        strict)
{
  cl_int errcode_ret;
  a_cl_command_queue = clCreateCommandQueue(context, device, properties, &errcode_ret);
  return confirm_cl_status(errcode_ret, hash, "cl_create_command_queue", strict);
}

Result cl_release_kernel(cl_kernel kernel, const std::string& hash, bool strict)
{
  cl_int ret = clReleaseKernel(kernel);
  return confirm_cl_status(ret, hash, "cl_release_kernel", strict);
}

Result cl_release_context(cl_context context, const std::string& hash, bool strict)
{
  cl_int ret = clReleaseContext(context);
  return confirm_cl_status(ret, hash, "cl_release_context", strict);
}

Result
cl_release_command_queue(cl_command_queue command_queue, const std::string& hash, bool strict)
{
  cl_int ret = clReleaseCommandQueue(command_queue);
  return confirm_cl_status(ret, hash, "cl_release_command_queue", strict);
}

Result cl_release_program(cl_program program, const std::string& hash, bool strict)
{
  cl_int ret = clReleaseProgram(program);
  return confirm_cl_status(ret, hash, "cl_release_program", strict);
}

Result cl_set_kernel_arg(cl_kernel&         kernel,
                         cl_uint            arg_index,
                         size_t             arg_size,
                         const void*        arg_value,
                         const std::string& hash,
                         bool               strict)
{

  if (kernel == nullptr)
  {
    std::stringstream errm;
    errm << "In cl_set_kernel_arg."
         << "Attempt to set kernel argument of uninitialised kernel."
         << "hash : `" << hash << "'";

    throw miog_error(errm.str());
  }

  cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);

  return confirm_cl_status(ret, hash, "cl_set_kernel_arg", strict);
}

Result cl_set_kernel_args(cl_kernel& kernel,
                          std::vector<std::pair<size_t, const void*>> arg_sizes_values,
                          const std::string& hash,
                          bool               strict)
{

  for (cl_uint arg_index = 0; arg_index < arg_sizes_values.size(); ++arg_index)
  {
    std::stringstream hashss;
    hashss << "cl_set_kernel_args with hash : `" << hash << "'. Attempting to set arg at index "
           << arg_index << ".";
    size_t      arg_size  = arg_sizes_values[arg_index].first;
    const void* arg_value = arg_sizes_values[arg_index].second;
    auto oclr = cl_set_kernel_arg(kernel, arg_index, arg_size, arg_value, hashss.str(), strict);

    if (oclr.fail())
    {
      return oclr;
    }
  }

  return {};
}

Result cl_flush(cl_command_queue command_queue, const std::string& hash, bool strict)
{
  cl_int ret = clFlush(command_queue);
  return confirm_cl_status(ret, hash, "cl_flush", strict);
}

Result cl_wait_for_events(cl_uint            num_events,
                          const cl_event*    event_list,
                          const std::string& hash,
                          bool               strict)
{
  cl_int ret = clWaitForEvents(num_events, event_list);
  return confirm_cl_status(ret, hash, "cl_wait_for_events", strict);
}

Result cl_set_command_queue_info(cl_command_queue      command_queue,
                                 cl_command_queue_info param_name,
                                 size_t                param_value_size,
                                 void*                 param_value,
                                 size_t*               param_value_size_ret,
                                 const std::string&    hash,
                                 bool                  strict)
{
  cl_int ret = clGetCommandQueueInfo(
    command_queue, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "cl_set_command_queue_info", strict);
}

Result cl_set_buffer(cl_mem&            a_cl_mem,
                     cl_context         context,
                     cl_mem_flags       flags,
                     size_t             size,
                     void*              host_ptr,
                     const std::string& hash,
                     bool               strict)
{
  cl_int errcode_ret;
  a_cl_mem = clCreateBuffer(context, flags, size, host_ptr, &errcode_ret);
  return confirm_cl_status(errcode_ret, hash, "cl_set_buffer", strict);
}

Result cl_enqueue_copy_buffer(cl_command_queue   command_queue,
                              cl_mem             src_buffer,
                              cl_mem             dst_buffer,
                              size_t             src_offset,
                              size_t             dst_offset,
                              size_t             cb,
                              cl_uint            num_events_in_wait_list,
                              const cl_event*    event_wait_list,
                              cl_event*          event,
                              const std::string& hash,
                              bool               strict)
{
  cl_int ret = clEnqueueCopyBuffer(command_queue,
                                   src_buffer,
                                   dst_buffer,
                                   src_offset,
                                   dst_offset,
                                   cb,
                                   num_events_in_wait_list,
                                   event_wait_list,
                                   event);
  return confirm_cl_status(ret, hash, "cl_enqueue_copy_buffer", strict);
}

Result cl_release_mem_object(cl_mem memobj, const std::string& hash, bool strict)
{
  cl_int ret = clReleaseMemObject(memobj);
  return confirm_cl_status(ret, hash, "cl_release_mem_object", strict);
}

Result cl_enqueue_ndrange_kernel(cl_command_queue   command_queue,
                                 cl_kernel          kernel,
                                 cl_uint            work_dim,
                                 const size_t*      global_work_offset,
                                 const size_t*      global_work_size,
                                 const size_t*      local_work_size,
                                 cl_uint            num_events_in_wait_list,
                                 const cl_event*    event_wait_list,
                                 cl_event*          event,
                                 const std::string& hash,
                                 bool               strict)
{
  cl_int ret = clEnqueueNDRangeKernel(command_queue,
                                      kernel,
                                      work_dim,
                                      global_work_offset,
                                      global_work_size,
                                      local_work_size,
                                      num_events_in_wait_list,
                                      event_wait_list,
                                      event);
  return confirm_cl_status(ret, hash, "cl_enqueue_ndrange_kernel", strict);
}

Result cl_set_buffer_from_command_queue(cl_mem&            a_cl_mem,
                                        cl_command_queue   command_queue,
                                        cl_mem_flags       flags,
                                        size_t             size,
                                        void*              host_ptr,
                                        const std::string& hash,
                                        bool               strict)
{

  cl_context context;

  auto oclr = cl_set_command_queue_info(command_queue,
                                        CL_QUEUE_CONTEXT,
                                        sizeof(cl_context),
                                        &context,
                                        nullptr,
                                        hash + " + (cl_set_buffer_from_command_queue)",
                                        strict);
  if (oclr.fail())
  {
    return oclr;
  }

  return cl_set_buffer(a_cl_mem,
                       context,
                       flags,
                       size,
                       host_ptr,
                       hash + "+ (cl_set_buffer_from_command_queue)",
                       strict);
}

Result cl_set_platform_ids(cl_uint            num_entries,
                           cl_platform_id*    platforms,
                           cl_uint*           num_platforms,
                           const std::string& hash,
                           bool               strict)
{
  cl_int ret = clGetPlatformIDs(num_entries, platforms, num_platforms);
  return confirm_cl_status(ret, hash, "cl_set_platform_ids", strict);
}

Result cl_set_context_from_type(
  cl_context&            a_cl_context,
  cl_context_properties* properties,
  cl_device_type         device_type,
  void (*pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void*              user_data,
  const std::string& hash,
  bool               strict)
{
  cl_int errcode;
  a_cl_context = clCreateContextFromType(properties, device_type, pfn_notify, user_data, &errcode);
  return confirm_cl_status(errcode, hash, "cl_set_context_from_type", strict);
}

Result cl_set_context_info(cl_context         context,
                           cl_context_info    param_name,
                           size_t             param_value_size,
                           void*              param_value,
                           size_t*            param_value_size_ret,
                           const std::string& hash,
                           bool               strict)
{
  cl_int ret =
    clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "cl_set_context_info", strict);
}

Result cl_set_device_info(cl_device_id       device,
                          cl_device_info     param_name,
                          size_t             param_value_size,
                          void*              param_value,
                          size_t*            param_value_size_ret,
                          const std::string& hash,
                          bool               strict)
{
  cl_int ret =
    clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "cl_set_device_info", strict);
}

Result cl_set_platform_info(cl_platform_id     platform,
                            cl_platform_info   param_name,
                            size_t             param_value_size,
                            void*              param_value,
                            size_t*            param_value_size_ret,
                            const std::string& hash,
                            bool               strict)
{
  cl_int ret =
    clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "cl_set_platform_info", strict);
}

Result cl_set_event_profiling_info(cl_event           event,
                                   cl_profiling_info  param_name,
                                   size_t             param_value_size,
                                   void*              param_value,
                                   size_t*            param_value_size_ret,
                                   const std::string& hash,
                                   bool               strict)
{
  cl_int ret =
    clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "cl_set_event_profiling_info", strict);
}

Result set_device_info_from_command_queue(cl_command_queue   command_queue,
                                          cl_device_info     param_name,
                                          size_t             param_value_size,
                                          void*              param_value,
                                          size_t*            param_value_size_ret,
                                          const std::string& hash,
                                          bool               strict)
{

  cl_device_id device;
  auto         oclr = cl_set_command_queue_info(command_queue,
                                        CL_QUEUE_DEVICE,
                                        sizeof(cl_device_id),
                                        &device,
                                        nullptr,
                                        hash + " + (in set_device_info_from_command_queue)",
                                        strict);
  if (oclr.fail())
  {
    return oclr;
  }

  return cl_set_device_info(device,
                            param_name,
                            param_value_size,
                            param_value,
                            param_value_size_ret,
                            hash + " + (in set_device_info_from_command_queue)",
                            strict);
}

Result cl_set_platform_info_from_command_queue(cl_command_queue   command_queue,
                                               cl_platform_info   param_name,
                                               size_t             param_value_size,
                                               void*              param_value,
                                               size_t*            param_value_size_ret,
                                               const std::string& hash,
                                               bool               strict)
{
  cl_platform_id platform;

  auto oclr = set_device_info_from_command_queue(
    command_queue,
    CL_DEVICE_PLATFORM,
    sizeof(cl_platform_id),
    &platform,
    NULL,
    "getting CL_DEVICE_PLATFORM in get_platform_info_from_command_queue",
    strict);
  if (oclr.fail())
    return oclr;

  return cl_set_platform_info(platform,
                              param_name,
                              param_value_size,
                              param_value,
                              param_value_size_ret,
                              hash + " + (in set_device_info_from_command_queue)",
                              strict);
}

Result cl_create_kernel(cl_kernel&         a_kernel,
                        cl_program         program,
                        const char*        kernel_name,
                        const std::string& hash,
                        bool               strict)
{
  cl_int errcode_ret;
  a_kernel = clCreateKernel(program, kernel_name, &errcode_ret);
  return confirm_cl_status(errcode_ret, hash, "cl_create_kernel", strict);
}

Result cl_create_program_with_source(cl_program&        a_cl_program,
                                     cl_context         context,
                                     cl_uint            count,
                                     const char**       strings,
                                     const size_t*      lengths,
                                     const std::string& hash,
                                     bool               strict)
{
  cl_int errcode_ret;
  a_cl_program = clCreateProgramWithSource(context, count, strings, lengths, &errcode_ret);
  return confirm_cl_status(errcode_ret, hash, "cl_create_program_with_source", strict);
}

Result cl_build_program(cl_program          program,
                        cl_uint             num_devices,
                        const cl_device_id* device_list,
                        const char*         options,
                        void (*pfn_notify)(cl_program, void* user_data),
                        void*              user_data,
                        owrite::Writer&    mowri,
                        const std::string& hash,
                        bool               strict)
{

  cl_int ret = clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);

  char   buffer[10240];
  size_t buffer_size;
  clGetProgramBuildInfo(
    program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &buffer_size);

  if (ret != CL_SUCCESS)
  {
    fprintf(stderr, "CL Compilation failed:\n%s", buffer);
    return confirm_cl_status(ret, hash, "cl_build_program", strict);
  }

  else
  {
    bool iswhitespace = true;
    for (size_t i = 0; i < buffer_size - 1; ++i)
    {
      if (std::isspace(buffer[i]) == false)
      {
        iswhitespace = false;
        break;
      }
    }

    if (iswhitespace == false)
    {
      std::stringstream ss_comp_warning;
      ss_comp_warning << "\n                  warning during compilation of "
                         "kernel, in cl_build_program:\n ";
      ss_comp_warning << ">>" << buffer << "<<";
      mowri << ss_comp_warning.str();
    }

    return confirm_cl_status(ret, hash, "cl_build_program", strict);
  }
}

Result cl_set_program_info(cl_program         program,
                           cl_program_info    param_name,
                           size_t             param_value_size,
                           void*              param_value,
                           size_t*            param_value_size_ret,
                           const std::string& hash,
                           bool               strict)
{
  cl_int ret =
    clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
  return confirm_cl_status(ret, hash, "Result cl_set_program_info", strict);
}

Result cl_enqueue_write_buffer(cl_command_queue   command_queue,
                               cl_mem             buffer,
                               cl_bool            blocking_write,
                               size_t             offset,
                               size_t             size,
                               const void*        ptr,
                               cl_uint            num_events_in_wait_list,
                               const cl_event*    event_wait_list,
                               cl_event*          event,
                               const std::string& hash,
                               bool               strict)
{
  cl_int ret = clEnqueueWriteBuffer(command_queue,
                                    buffer,
                                    blocking_write,
                                    offset,
                                    size,
                                    ptr,
                                    num_events_in_wait_list,
                                    event_wait_list,
                                    event);
  return confirm_cl_status(ret, hash, "cl_enqueue_write_buffer", strict);
}

Result cl_enqueue_read_buffer(cl_command_queue   command_queue,
                              cl_mem             buffer,
                              cl_bool            blocking_read,
                              size_t             offset,
                              size_t             cb,
                              void*              ptr,
                              cl_uint            num_events_in_wait_list,
                              const cl_event*    event_wait_list,
                              cl_event*          event,
                              const std::string& hash,
                              bool               strict)
{
  cl_int ret = clEnqueueReadBuffer(command_queue,
                                   buffer,
                                   blocking_read,
                                   offset,
                                   cb,
                                   ptr,
                                   num_events_in_wait_list,
                                   event_wait_list,
                                   event);
  return confirm_cl_status(ret, hash, "cl_enqueue_read_buffer", strict);
}

Result cl_set_platform_etc(cl_platform_id&    platform,
                           cl_uint&           num_platforms,
                           cl_context&        context,
                           cl_device_id&      device_id_to_use,
                           owrite::Writer&    mowri,
                           const std::string& hash,
                           bool               strict)
{

  // Get the platform(s)

  std::vector<cl_platform_id> platforms(10);
  auto                        oclr = cl_set_platform_ids(
    platforms.size(), &platforms[0], &num_platforms, hash + " in cl_set_platform_etc", strict);
  if (oclr.fail())
    return oclr;

  bool multi_platforms_resolved = false;
  if (num_platforms > 1)
  {
    std::stringstream errm;
    errm << num_platforms << " platforms detected\n-----------------";
    for (size_t i = 0; i < num_platforms; ++i)
    {
      OpenCLPlatformInfo platinfo(platforms[i]);
      if (platinfo.vendor.find("NVIDIA") != std::string::npos ||
          platinfo.vendor.find("vidia") != std::string::npos)
      {
        platform                 = platforms[i];
        multi_platforms_resolved = true;
      }
      errm << platinfo.get_string();
      errm << "-----------------\n";
    }

    if (multi_platforms_resolved == false)
    {
      std::string errmessage = errm.str();
      throw miog_error(errmessage);
    }
  }

  else
  {
    platform = platforms[0];
  }

  // Create context
  cl_context_properties cps[3] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
  cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
  oclr                          = cl_set_context_from_type(context,
                                  cprops,
                                  CL_DEVICE_TYPE_GPU,
                                  nullptr,
                                  nullptr,
                                  hash + " in cl_set_platform_etc...",
                                  strict);
  if (oclr.fail())
    return oclr;

  // Query the number of devices
  int deviceListSize;
  oclr = cl_set_context_info(context,
                             CL_CONTEXT_NUM_DEVICES,
                             sizeof(int),
                             &deviceListSize,
                             nullptr,
                             hash + " getting deviceListSize",
                             strict);
  if (oclr.fail())
    return oclr;

  if (deviceListSize == 0)
  {
    throw miog_error("There are no devices detected. \nSpecifically, using clGetContextInfo "
                     "with "
                     "CL_CONTEX_NUM_DEVICES as the flag returns 0. \nThis error is being "
                     "thrown "
                     "from cl_set_platform_etc in oclutil.cpp. Please have a look, it "
                     "seems "
                     "like the current boilerplate can't figure out your setup.");
  }

  std::vector<cl_device_id> devices(deviceListSize);

  // Get the devices
  oclr = cl_set_context_info(context,
                             CL_CONTEXT_DEVICES,
                             deviceListSize * sizeof(cl_device_id),
                             devices.data(),
                             nullptr,
                             hash + " getting the devices",
                             strict);
  if (oclr.fail())
    return oclr;

  char    deviceName[100];
  cl_uint max_compute_units;

  // Go through the devices and see how many compute
  // units each has, storing the best along the way
  cl_uint      max_max_compute_units = 0;
  char         bestDeviceName[100];
  cl_device_id bestDeviceId = 0;

  std::string device_compute_unit_count_string = "The following devices were detected: \n";

  for (int i = 0; i < deviceListSize; ++i)
  {
    /* get name and number of compute units of the device whose id is devices[i]
     */

    oclr = cl_set_device_info(devices[i],
                              CL_DEVICE_NAME,
                              sizeof(deviceName),
                              deviceName,
                              nullptr,
                              hash + " getting CL_DEVICE_NAME",
                              strict);
    if (oclr.fail())
      return oclr;

    oclr = cl_set_device_info(devices[i],
                              CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(cl_uint),
                              &max_compute_units,
                              nullptr,
                              " getting CL_DEVICE_MAX_COMPUTE_UNITS",
                              strict);
    if (oclr.fail())
      return oclr;

    if (max_compute_units > max_max_compute_units)
    {
      max_max_compute_units = max_compute_units;
      oclr                  = cl_set_device_info(devices[i],
                                CL_DEVICE_NAME,
                                sizeof(bestDeviceName),
                                bestDeviceName,
                                nullptr,
                                hash + " getting best devices CL_DEVICE_NAME",
                                strict);
      if (oclr.fail())
        return oclr;

      bestDeviceId = devices[i];
    }

    device_compute_unit_count_string += deviceName;
    device_compute_unit_count_string += ", which as ";
    device_compute_unit_count_string += std::to_string(max_compute_units);
    device_compute_unit_count_string += " compute units\n";
  }

  bool only_good_hardware = true;
  if (only_good_hardware == true)
  {
    // getting vendor
    std::string info_st("");
    info_st.resize(2048, '-');
    size_t info_size;

    oclr = cl_set_platform_info(platform,
                                CL_PLATFORM_VENDOR,
                                info_st.size(),
                                &info_st[0],
                                &info_size,
                                hash + " obtaining CL_PLATFORM_VENDOR",
                                strict);
    if (oclr.fail())
      return oclr;

    std::string platform_vendor = info_st.substr(0, info_size);
    // assuming that if it's Nvidia, it's correct
    if (platform_vendor.find("vidia") != std::string::npos ||
        platform_vendor.find("NVIDIA") != std::string::npos)
    {
    }

    // if not Nvidia, and fewer than 40 compute units
    else if (max_max_compute_units < 40)
    {
      std::string errm = device_compute_unit_count_string;
      errm += "As this is less than 40, an error is being thrown. \nIf you "
              "wish to use a device "
              "with fewer than 40 CUs, please make changes here (in "
              "oclutil.cpp)";
      throw miog_error(errm);
    }
  }

  mowri << "Will use device with name " << bestDeviceName << ", which reports to have "
        << max_max_compute_units << " CUs. \nTo use a different device, consider modifying "
                                    "cl_set_platform_etc in "
                                    "oclutil.cpp (or write custom OpenCL boilerplate)."
        << Endl;
  device_id_to_use = bestDeviceId;

  return {};
}

Result cl_set_program_and_kernel(const cl_command_queue& command_queue,
                                 const std::string&      kernel_string,
                                 const std::string&      kernel_function_name,
                                 cl_program&             program,
                                 cl_kernel&              kernel,
                                 owrite::Writer&         mowri,
                                 bool                    strict)
{

  cl_context context;
  auto       oclr = cl_set_command_queue_info(command_queue,
                                        CL_QUEUE_CONTEXT,
                                        sizeof(cl_context),
                                        &context,
                                        NULL,
                                        "getting context from queue in set_program_and_kernel",
                                        strict);
  if (oclr.fail())
    return oclr;

  cl_device_id device_id_to_use;
  oclr = cl_set_command_queue_info(command_queue,
                                   CL_QUEUE_DEVICE,
                                   sizeof(cl_device_id),
                                   &device_id_to_use,
                                   NULL,
                                   "getting device id from queue in set_program_and_kernel",
                                   strict);
  if (oclr.fail())
    return oclr;

  auto kernel_cstr = kernel_string.c_str();

  auto kernel_string_size = kernel_string.size();

  oclr = cl_create_program_with_source(program,
                                       context,
                                       1,
                                       &kernel_cstr,
                                       &kernel_string_size,
                                       "creating program in set_program_and_kernel",
                                       strict);
  if (oclr.fail())
    return oclr;

  // To generate isa code, you'll need to add something like
  //-save-temps= + "/some/path/"
  // to the following string
  std::string buildOptions_11 = "-cl-std=CL2.0  -Werror";
  auto        buildOptions    = buildOptions_11.c_str();

  oclr = cl_build_program(program,
                          1,
                          &device_id_to_use,
                          buildOptions,
                          NULL,
                          NULL,
                          mowri,
                          "building program in set_program_and_kernel",
                          strict);
  if (oclr.fail())
    return oclr;

  // the store as binary option removed *here*

  oclr = cl_create_kernel(kernel,
                          program,
                          kernel_function_name.c_str(),
                          "getting kernel in set_program_and_kernel",
                          strict);
  return oclr;
}

SafeClMem::SafeClMem(const std::string& hash_) : clmem(nullptr), hash(hash_) {}

SafeClMem::~SafeClMem()
{
  if (clmem != nullptr)
  {
    bool strict = true;
    auto oclr   = oclutil::cl_release_mem_object(clmem, hash, strict);
  }
}

// properties = CL_QUEUE_PROFILING_ENABLE :
// create an inorder command queue with
// profiling enabled. Profiling is enabled
// so that we can get
// start and end times for kernels
Result cl_auto_set_command_queue(cl_command_queue&           a_cl_command_queue,
                                 owrite::Writer&             mowri,
                                 cl_command_queue_properties properties,
                                 const std::string&          hash,
                                 bool                        strict)
{

  cl_platform_id platform = nullptr;
  cl_uint        num_platforms;
  cl_context     context;
  cl_device_id   device_id_to_use;

  auto oclr = oclutil::cl_set_platform_etc(platform,
                                           num_platforms,
                                           context,
                                           device_id_to_use,
                                           mowri,
                                           hash + "from cl_auto_set_command_queue",
                                           strict);
  if (oclr.fail())
    return oclr;

  return cl_set_command_queue(a_cl_command_queue,
                              context,
                              device_id_to_use,
                              properties,
                              hash + "from cl_auto_set_command_queue",
                              strict);
}

CommandQueueInContext::CommandQueueInContext(owrite::Writer& mowri, const std::string& hash_)
  : hash(hash_)
{
  bool strict = true;
  cl_auto_set_command_queue(
    command_queue, mowri, CL_QUEUE_PROFILING_ENABLE, "CommandQueueInContext constructor", strict);
}

CommandQueueInContext::~CommandQueueInContext()
{
  bool strict = true;
  if (command_queue != nullptr)
  {
    cl_context context;
    cl_set_command_queue_info(command_queue,
                              CL_QUEUE_CONTEXT,
                              sizeof(cl_context),
                              &context,
                              nullptr,
                              hash + " + (CommandQueueInContext destuctor)",
                              strict);
    cl_release_context(context, "in destructor of CommandQueueInContext", strict);
    cl_release_command_queue(command_queue, "in destructor of CommandQueueInContext", strict);
  }
}

OpenCLPlatformInfo::OpenCLPlatformInfo(cl_platform_id platform_id)
{

  bool        strict = true;
  std::string info_st("");
  info_st.resize(2048, '-');
  size_t info_size;

  cl_set_platform_info(platform_id,
                       CL_PLATFORM_PROFILE,
                       info_st.size(),
                       &info_st[0],
                       &info_size,
                       "getting CL_PLATFORM_PROFILE for OpenCLPlatformInfo",
                       strict);
  profile = info_st.substr(0, info_size - 1);

  cl_set_platform_info(platform_id,
                       CL_PLATFORM_VERSION,
                       info_st.size(),
                       &info_st[0],
                       &info_size,
                       "getting CL_PLATFORM_VERSION for OpenCLPlatformInfo",
                       strict);
  version = info_st.substr(0, info_size - 1);

  cl_set_platform_info(platform_id,
                       CL_PLATFORM_NAME,
                       info_st.size(),
                       &info_st[0],
                       &info_size,
                       "getting CL_PLATFORM_NAME for OpenCLPlatformInfo",
                       strict);
  name = info_st.substr(0, info_size - 1);

  cl_set_platform_info(platform_id,
                       CL_PLATFORM_VENDOR,
                       info_st.size(),
                       &info_st[0],
                       &info_size,
                       "getting CL_PLATFORM_VENDOR for OpenCLPlatformInfo",
                       strict);
  vendor = info_st.substr(0, info_size - 1);
}

// taken from MIOpen

std::string GetDeviceNameFromMap(const std::string& name)
{

  static std::map<std::string, std::string> device_name_map = {
    {"Ellesmere", fiji_string},
    {"Baffin", fiji_string},
    {"RacerX", fiji_string},
    {"Polaris10", fiji_string},
    {"Polaris11", fiji_string},
    {"Tonga", fiji_string},
    {"Fiji", fiji_string},
    {"gfx800", fiji_string},
    {"gfx802", fiji_string},
    {"gfx803", fiji_string},
    {"gfx804", fiji_string},
    {"Vega10", vega_string},
    {"gfx900", vega_string},
    {"gfx901", vega_string},
  };

  auto device_name_iterator = device_name_map.find(name);
  if (device_name_iterator != device_name_map.end())
  {
    return device_name_iterator->second;
  }
  else
  {
    return name;
  }
}

DevInfo::DevInfo(const cl_command_queue& command_queue)
{
  std::string info_st("");
  info_st.resize(2048, '-');
  size_t info_size;

  cl_bool  a_bool;
  cl_ulong a_ulong;
  cl_uint  a_uint;

  bool strict = true;

  cl_platform_id platform;
  set_device_info_from_command_queue(
    command_queue,
    CL_DEVICE_PLATFORM,
    sizeof(cl_platform_id),
    &platform,
    NULL,
    "getting CL_DEVICE_PLATFORM in get_platform_info_from_command_queue",
    strict);

  platinfo = OpenCLPlatformInfo(platform);

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_NAME,
                                              info_st.size(),
                                              &info_st[0],
                                              &info_size,
                                              "obtaining CL_DEVICE_NAME",
                                              strict);
  device_name = info_st.substr(0, info_size - 1);
  device_name = GetDeviceNameFromMap(device_name);

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_AVAILABLE,
                                              sizeof(cl_bool),
                                              &a_bool,
                                              NULL,
                                              "obtaining CL_DEVICE_AVAILABLE",
                                              strict);
  device_available = a_bool;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_GLOBAL_MEM_SIZE,
                                              sizeof(cl_ulong),
                                              &a_ulong,
                                              NULL,
                                              "obtaining CL_DEVICE_GLOBAL_MEM_SIZE",
                                              strict);
  device_global_mem_size = a_ulong;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_LOCAL_MEM_SIZE,
                                              sizeof(cl_ulong),
                                              &a_ulong,
                                              NULL,
                                              "obtaining CL_DEVICE_LOCAL_MEM_SIZE",
                                              strict);
  device_local_mem_size = a_ulong;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                              sizeof(cl_uint),
                                              &a_uint,
                                              NULL,
                                              "obtaining CL_DEVICE_MAX_CLOCK_FREQUENCY",
                                              strict);
  device_max_clock_frequency = a_uint;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_MAX_COMPUTE_UNITS,
                                              sizeof(cl_uint),
                                              &a_uint,
                                              NULL,
                                              "obtaining CL_DEVICE_MAX_COMPUTE_UNITS",
                                              strict);
  device_max_compute_units = a_uint;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                              sizeof(cl_ulong),
                                              &a_ulong,
                                              NULL,
                                              "obtaining CL_DEVICE_MAX_WORK_GROUP_SIZE",
                                              strict);
  device_max_work_group_size = a_ulong;

  // oclutil::set_device_info_from_command_queue(command_queue,
  // CL_DEVICE_MAX_WORK_GROUP_SIZE,
  // sizeof(cl_ulong),
  //&a_ulong,
  // NULL,
  //"obtaining CL_DEVICE_MAX_WORK_GROUP_SIZE",
  // strict);
  // device_max_work_group_size = a_ulong;

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DEVICE_VERSION,
                                              info_st.size(),
                                              &info_st[0],
                                              &info_size,
                                              "obtaining CL_DEVICE_VERSION",
                                              strict);
  device_version = info_st.substr(0, info_size - 1);

  oclutil::set_device_info_from_command_queue(command_queue,
                                              CL_DRIVER_VERSION,
                                              info_st.size(),
                                              &info_st[0],
                                              &info_size,
                                              "obtaining CL_DRIVER_VERSION",
                                              strict);
  driver_version = info_st.substr(0, info_size - 1);

  if (platinfo.vendor.find("vidia") != std::string::npos ||
      platinfo.vendor.find("NVIDIA") != std::string::npos)
  {
    wg_atom_size = 32;
  }

  else if (platinfo.vendor.find("Advanced Micro") != std::string::npos ||
           platinfo.vendor.find("Advanced Micro") != std::string::npos ||
           platinfo.vendor.find("AMD") != std::string::npos)
  {
    if (device_name == vega_string)
    {
      wg_atom_size = 32;
    }

    else
    {
      wg_atom_size = 64;
    }
  }

  else
  {
    wg_atom_size = 32;
    throw miog_error(" has not been tested on any platform from vendor " + platinfo.vendor +
                     " yet. Are you sure you want to try "
                     "this ? If so, remove error message "
                     "hyiar");
  }

  // setting identifier
  bool fancy_identifier = false;
  if (fancy_identifier == true)
  {
    std::stringstream idss;
    idss << "";
    for (const std::string& st : {device_name, device_version, driver_version})
    {
      for (char c : st)
      {

        if ((bool)std::isalnum(c) == true)
        {
          idss << c;
        }
        else if (c == '.')
        {
          idss << 'p';
        }
      }
    }

    identifier = idss.str();
  }

  else
  {
    // Do as in MIOpen, just use the device_name
    identifier = device_name;
  }
}

DevInfo::DevInfo() { device_name = "unknown_default_constructed"; }

std::string OpenCLPlatformInfo::get_string() const
{
  std::stringstream ss;
  ss << "\n";
  ss << "platform profile : " << profile << "\n";
  ss << "platform vendor : " << vendor << "\n";
  ss << "platform version : " << version << "\n";
  ss << "platform name : " << name << "\n";
  return ss.str();
}

std::string DevInfo::get_string() const
{
  std::stringstream ss;
  ss << platinfo.get_string();
  ss << "device name : " << device_name << "\n";
  ss << "device version : " << device_version << "\n";
  ss << "driver version : " << driver_version << "\n";
  ss << "device_available : " << device_available << "\n";
  ss << "device_global_mem_size : " << device_global_mem_size << "\n";
  ss << "device_max_clock_frequency : " << device_max_clock_frequency << "\n";
  ss << "device_max_compute_units : " << device_max_compute_units << "\n";
  ss << "device_max_work_group_size : " << device_max_work_group_size << "\n";
  ss << "(identifier) : " << identifier << "\n";
  ss << "\n";

  return ss.str();
}
}
}
