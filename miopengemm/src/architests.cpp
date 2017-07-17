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

Stat::Stat(const oclutil::DevInfo& devinfo,
           const DerivedParams&    dp,
           const Geometry&         gg,
           const HyPas&            hp)
{
  (void)hp;

  std::stringstream status_ss;

  // check -1 : macro tile not too large
  if (dp.main_n_work_items_per_workgroup > devinfo.device_max_work_group_size)
  {
    status_ss << "n_work_items_per_workgroup > CL_DEVICE_MAX_WORK_GROUP_SIZE, ( "
              << dp.main_n_work_items_per_workgroup << " > " << devinfo.device_max_work_group_size
              << " ) : cannot compile this kernel to this architecture \n";
  }

  // check 0 : LDS
  size_t LDS_required =
    gg.derived.float_size_bytes * ((dp.at(Mat::E::A).main_n_elements_in_padded_unroll +
                                    dp.at(Mat::E::B).main_n_elements_in_padded_unroll));

  if (LDS_required >= devinfo.device_local_mem_size)  // max_LDS_bytes)
  {
    status_ss << "LDS_required (" << LDS_required << ")  >= max_LDS_bytes ("
              << devinfo.device_local_mem_size << ") \n";
  }

  msg     = status_ss.str();
  is_good = (msg == "");
}
}
}
