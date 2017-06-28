/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#include <sstream>
#include <miopengemm/architests.hpp>
#include <miopengemm/openclutil.hpp>

namespace MIOpenGEMM
{
namespace architests
{

std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue command_queue,
                                                          const derivedparams::DerivedParams& dp,
                                                          const Geometry&                     gg,
                                                          const hyperparams::HyperParams&     hp)
{
  (void)hp;

  std::stringstream status_ss;
  size_t            max_work_group_size;

  openclutil::set_device_info_from_command_queue(
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
  openclutil::set_device_info_from_command_queue(
    command_queue,
    CL_DEVICE_LOCAL_MEM_SIZE,
    sizeof(size_t),
    &max_LDS_bytes,
    nullptr,
    "getting CL_DEVICE_LOCAL_MEM_SIZE in artcitecture specfic stests",
    true);

  unsigned LDS_required =
    gg.derived.float_size_bytes * ((dp.at(nsHP::matA).main_n_elements_in_padded_unroll +
                                    dp.at(nsHP::matB).main_n_elements_in_padded_unroll));

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
