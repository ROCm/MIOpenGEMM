/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEVDIVA_HPP
#define GUARD_MIOPENGEMM_DEVDIVA_HPP

#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>
#include <array>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFloat>
class Diva
{

  public:
  Diva(Geometry                     gg_,
       Offsets                      toff_,
       const TFloat*                a_,
       const TFloat*                b_,
       const TFloat*                c_,
       owrite::Writer& mowri_);

  Diva(Geometry gg_, Offsets toff_, owrite::Writer& mowri_);



  void benchgemm(const std::vector<std::string>& hyperstrings,
                 size_t                          max_number_of_runs,
                 double                          max_time_per_kernel);

  Solution find(const FindParams& find_params, std::string constraints_string);



  void accuracy_test(const std::string& hyperstring, const TFloat* c_true_for_test);

  ///////////////////////////////////

  private:
  Geometry gg;
  Offsets  toff;

  // a, b and c cpu memories.
  std::vector<const TFloat*> cpu_mem;

  // used when no pointer constructor is used.
  std::array<std::vector<TFloat>, Mat::E::N> __cpu_mem;

  std::vector<TFloat>          c_copy;
  std::vector<TFloat>          c_for_cpu_compute;
  owrite::Writer& mowri;

  oclutil::CommandQueueInContext tgcq;

  // a, b, c and workspace, gpu memories.
  std::vector<oclutil::SafeClMem> gpu_safemem;

  // sizes of a, b, c and workspace gpu memories.
  std::vector<size_t> mem_size;

  // read write permissions of gpu data TODO : move to enums.cpp
  std::vector<cl_mem_flags> rw_perms;

  //size_t get_mat_memsize(Mem::E emem);

  size_t get_workspace_memsize();

  void opencl_memory_initialise();

  void initialise_cpu_mem_from_scratch();

  void initialise_cpu_mem(const TFloat* a_, const TFloat* b_, const TFloat* c_);

  void initialise_common();

  // delegator constructor.
  Diva(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, long);
};


}
}

#endif
