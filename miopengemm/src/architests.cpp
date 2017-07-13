/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <sstream>
#include <miopengemm/architests.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{
namespace architests
{

std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue command_queue,
                                                          const DerivedParams& dp,
                                                          const Geometry&                     gg,
                                                          const HyPas&     hp)
{
  (void)hp;

  std::stringstream status_ss;
  size_t            max_work_group_size;

  oclutil::set_device_info_from_command_queue(
    command_queue,
    CL_DEVICE_MAX_WORK_GROUP_SIZE,
    sizeof(size_t),
    &max_work_group_size,
    nullptr,
    "getting CL_DEVICE_MAX_WORK_GROUP_SIZE in artcitecture specfic stests",
    true);

  // check -1 : macro tile not too large
  if (dp.main_n_work_items_per_workgroup > max_work_group_size)
  {
    status_ss << "n_work_items_per_workgroup > CL_DEVICE_MAX_WORK_GROUP_SIZE, ( "
              << dp.main_n_work_items_per_workgroup << " > " << max_work_group_size
              << " ) : cannot compile this kernel to this architecture \n";
  }

  // check 0 : LDS
  cl_long max_LDS_bytes;
  oclutil::set_device_info_from_command_queue(
    command_queue,
    CL_DEVICE_LOCAL_MEM_SIZE,
    sizeof(size_t),
    &max_LDS_bytes,
    nullptr,
    "getting CL_DEVICE_LOCAL_MEM_SIZE in artcitecture specfic stests",
    true);

  size_t LDS_required =
    gg.derived.float_size_bytes * ((dp.at(Mat::E::A).main_n_elements_in_padded_unroll +
                                    dp.at(Mat::E::B).main_n_elements_in_padded_unroll));

  if (LDS_required >= max_LDS_bytes)
  {
    status_ss << "LDS_required (" << LDS_required << ")  >= max_LDS_bytes (" << max_LDS_bytes
              << ") \n";
  }

  std::string status_string = status_ss.str();
  return std::make_tuple(status_string.compare("") == 0, status_string);
}
}
}
