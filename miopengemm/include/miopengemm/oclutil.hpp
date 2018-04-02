/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_OPENCLUTIL_HPP
#define GUARD_MIOPENGEMM_OPENCLUTIL_HPP

#include <limits>
#include <tuple>
#include <miopengemm/hint.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/platform.hpp>

namespace MIOpenGEMM
{

namespace oclutil
{

cl_mem get_copy(cl_command_queue cq, cl_mem c, size_t c_nbytes, const std::string& hash);

std::string get_device_name(const cl_device_id& device_id, const std::string& hash, bool strict);

class Result
{
  public:
  cl_int      success;
  std::string message;
  Result() : success(CL_SUCCESS), message("") {}
  Result(cl_int success_, std::string message_) : success(success_), message(message_) {}
  bool fail() { return success != CL_SUCCESS; }
};

Result
confirm_cl_status(cl_int ret, const std::string& hash, const std::string& function, bool strict);

Result cl_set_command_queue(cl_command_queue&           a_cl_command_queue,
                            cl_context                  context,
                            cl_device_id                device,
                            cl_command_queue_properties properties,
                            const std::string&          hash,
                            bool                        strict);

Result cl_release_event(cl_event event, const std::string& hash, bool strict);

Result cl_release_kernel(cl_kernel kernel, const std::string& hash, bool strict);

Result cl_release_context(cl_context context, const std::string& hash, bool strict);

Result
cl_release_command_queue(cl_command_queue command_queue, const std::string& hash, bool strict);

Result cl_release_program(cl_program program, const std::string& hash, bool strict);

Result cl_set_kernel_arg(cl_kernel&         kernel,
                         cl_uint            arg_index,
                         size_t             arg_size,
                         const void*        arg_value,
                         const std::string& hash,
                         bool               strict);

Result cl_set_kernel_args(cl_kernel& kernel,
                          const std::vector<std::pair<size_t, const void*>>& arg_sizes_values,
                          const std::string& hash,
                          bool               strict);

Result cl_flush(cl_command_queue command_queue, const std::string& hash, bool strict);
Result cl_wait_for_events(cl_uint            num_events,
                          const cl_event*    event_list,
                          const std::string& hash,
                          bool               strict);

Result cl_set_command_queue_info(cl_command_queue      command_queue,
                                 cl_command_queue_info param_name,
                                 size_t                param_value_size,
                                 void*                 param_value,
                                 size_t*               param_value_size_ret,
                                 const std::string&    hash,
                                 bool                  strict);

Result cl_set_program_build_info(cl_program            program,
                                 cl_device_id          device,
                                 cl_program_build_info param_name,
                                 size_t                param_value_size,
                                 void*                 param_value,
                                 size_t*               param_value_size_ret,
                                 const std::string&    hash,
                                 bool                  strict);

Result cl_set_buffer(cl_mem&            a_cl_mem,
                     cl_context         context,
                     cl_mem_flags       flags,
                     size_t             size,
                     void*              host_ptr,
                     const std::string& hash,
                     bool               strict);

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
                              bool               strict);

Result cl_release_mem_object(cl_mem memobj, const std::string& hash, bool strict);

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
                                 bool               strict);

Result cl_set_platform_ids(cl_uint            num_entries,
                           cl_platform_id*    platforms,
                           cl_uint*           num_platforms,
                           const std::string& hash,
                           bool               strict);

Result cl_set_context_from_type(
  cl_context&            a_cl_context,
  cl_context_properties* properties,
  cl_device_type         device_type,
  void (*pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void*              user_data,
  const std::string& hash,
  bool               strict);

Result cl_set_context_info(cl_context         context,
                           cl_context_info    param_name,
                           size_t             param_value_size,
                           void*              param_value,
                           size_t*            param_value_size_ret,
                           const std::string& hash,
                           bool               strict);

Result cl_set_device_info(cl_device_id       device,
                          cl_device_info     param_name,
                          size_t             param_value_size,
                          void*              param_value,
                          size_t*            param_value_size_ret,
                          const std::string& hash,
                          bool               strict);

Result cl_set_platform_info(cl_platform_id     platform,
                            cl_platform_info   param_name,
                            size_t             param_value_size,
                            void*              param_value,
                            size_t*            param_value_size_ret,
                            const std::string& hash,
                            bool               strict);

Result cl_set_event_profiling_info(cl_event           event,
                                   cl_profiling_info  param_name,
                                   size_t             param_value_size,
                                   void*              param_value,
                                   size_t*            param_value_size_ret,
                                   const std::string& hash,
                                   bool               strict);

Result set_device_info_from_command_queue(cl_command_queue   command_queue,
                                          cl_device_info     param_name,
                                          size_t             param_value_size,
                                          void*              param_value,
                                          size_t*            param_value_size_ret,
                                          const std::string& hash,
                                          bool               strict);

Result cl_set_platform_info_from_command_queue(cl_command_queue   command_queue,
                                               cl_platform_info   param_name,
                                               size_t             param_value_size,
                                               void*              param_value,
                                               size_t*            param_value_size_ret,
                                               const std::string& hash,
                                               bool               strict);

Result cl_create_kernel(cl_kernel&         a_kernel,
                        cl_program         program,
                        const char*        kernel_name,
                        const std::string& hash,
                        bool               strict);

Result cl_create_program_with_source(cl_program&        a_cl_program,
                                     cl_context         context,
                                     cl_uint            count,
                                     const char**       strings,
                                     const size_t*      lengths,
                                     const std::string& hash,
                                     bool               strict);

Result cl_build_program(cl_program          program,
                        cl_uint             num_devices,
                        const cl_device_id* device_list,
                        const char*         options,
                        void (*pfn_notify)(cl_program, void* user_data),
                        void*              user_data,
                        owrite::Writer&    mowri,
                        const std::string& hash,
                        bool               strict);

Result cl_set_program_info(cl_program         program,
                           cl_program_info    param_name,
                           size_t             param_value_size,
                           void*              param_value,
                           size_t*            param_value_size_ret,
                           const std::string& hash,
                           bool               strict);

Result cl_set_buffer_from_command_queue(cl_mem&            a_cl_mem,
                                        cl_command_queue   command_queue,
                                        cl_mem_flags       flags,
                                        size_t             size,
                                        void*              host_ptr,
                                        const std::string& hash,
                                        bool               strict);

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
                               bool               strict);

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
                              bool               strict);

Result cl_set_platform_etc(cl_platform_id&    platform,
                           cl_uint&           num_platforms,
                           cl_context&        context,
                           cl_device_id&      device_id_to_use,
                           owrite::Writer&    mowri,
                           const CLHint&      xhint,
                           const std::string& hash,
                           bool               strict);

Result cl_set_program(const cl_context&   context,
                      const cl_device_id& device_id_to_use,
                      const std::string&  kernel_string,
                      // const std::string&  kernel_function_name,
                      cl_program& program,
                      // cl_kernel&          kernel,
                      const std::string& build_options,
                      owrite::Writer&    mowri,
                      bool               strict);

Result cl_set_context_and_device_from_command_queue(const cl_command_queue& command_queue,
                                                    cl_context&             context,
                                                    cl_device_id&           device_id,
                                                    owrite::Writer&         mowri,
                                                    bool                    strict);

Result cl_auto_set_command_queue(cl_command_queue&           a_cl_command_queue,
                                 owrite::Writer&             mowri,
                                 cl_command_queue_properties properties,
                                 const CLHint&               xhint,
                                 const std::string&          hash,
                                 bool                        strict);

class SafeClMem
{
  public:
  cl_mem      clmem;
  std::string hash;
  SafeClMem(const std::string& hash);

  SafeClMem(const SafeClMem&) = default;  // TODO : is this ok?

  ~SafeClMem();
};

class SafeClEvent
{
  public:
  cl_event    clevent;
  std::string hash;
  SafeClEvent(const std::string& hash);
  SafeClEvent() : SafeClEvent("safe event") {}

  ~SafeClEvent();
};

class CommandQueueInContext
{
  public:
  cl_command_queue command_queue;
  std::string      hash;
  CommandQueueInContext(owrite::Writer&             mowri,
                        cl_command_queue_properties properties,
                        const CLHint&               xhint,
                        const std::string&          hash);
  ~CommandQueueInContext();
};

class OpenCLPlatformInfo
{
  public:
  std::string profile;
  std::string version;
  std::string name;
  std::string vendor;

  OpenCLPlatformInfo(cl_platform_id platform_id);
  OpenCLPlatformInfo() = default;
  std::string get_string() const;
};

class DevInfo
{

  private:
  void initialise();

  cl_device_id device;

  public:
  std::string device_name    = "unknown";
  std::string device_version = "unknown";
  std::string driver_version = "unknown";
  std::string identifier     = "unknown";

  bool   device_available = false;
  size_t device_global_mem_size{0};
  size_t device_local_mem_size{0};
  size_t device_max_clock_frequency{0};
  size_t device_max_compute_units{0};
  size_t device_max_work_group_size{0};
  size_t wg_atom_size{0};

  std::string get_string() const;
  DevInfo(const cl_command_queue& command_queue);
  DevInfo(const cl_device_id& device);
  DevInfo(const CLHint& hint, owrite::Writer& mowri);
  // hack needed temporarily for get_fiji_device.
  DevInfo(const std::string& identifier, const std::string& device_name, size_t wg_atom_size);
};

DevInfo get_fiji_devinfo();

DevInfo get_vega_devinfo();
}
}

#endif
