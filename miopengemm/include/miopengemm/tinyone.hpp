/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEVDIVA_HPP
#define GUARD_MIOPENGEMM_DEVDIVA_HPP

#include <array>
#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFloat>
class TinyOne
{

  public:
  TinyOne(Geometry        gg_,
          Offsets         toff_,
          const TFloat*   a_,
          const TFloat*   b_,
          const TFloat*   c_,
          owrite::Writer& mowri_,
          const CLHint&   xhint);

  TinyOne(Geometry gg_,
          Offsets  toff_,
          std::array<const TFloat*, Mat::E::N>,
          owrite::Writer& mowri_,
          const CLHint&   xhint);

  TinyOne(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint);

  std::vector<std::vector<double>> benchgemm(const std::vector<HyPas>& hps, const Halt&);

  Solution find1(const FindParams& find_params, const Constraints& constraints);

  void accuracy_test(const HyPas& hp);  //, const TFloat* c_true_for_test);

  private:
  Geometry gg;
  Offsets  toff;

  // a, b and c cpu memories.
  std::vector<const TFloat*> cpu_mem;

  // used when no pointer constructor is used.
  std::array<std::vector<TFloat>, Mat::E::N> __cpu_mem;

  std::vector<TFloat> c_copy;
  //  std::vector<TFloat> c_for_cpu_compute;
  owrite::Writer& mowri;

  oclutil::CommandQueueInContext tgcq;

  // a, b, c and workspace, gpu memories.
  std::vector<oclutil::SafeClMem> gpu_safemem;

  // sizes of a, b, c and workspace gpu memories.
  std::vector<size_t> mem_size;

  // read write permissions of gpu data
  std::vector<cl_mem_flags> rw_perms;

  std::unique_ptr<TinyZero> up_jinx;

  size_t get_workspace_memsize();

  void opencl_memory_initialise();

  void initialise_cpu_mem_from_scratch();

  void initialise_cpu_mem(const TFloat* a_, const TFloat* b_, const TFloat* c_);

  void initialise_common();

  // delegator constructor.
  TinyOne(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint, long);
};
}
}

#endif
