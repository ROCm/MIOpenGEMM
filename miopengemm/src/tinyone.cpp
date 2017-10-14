/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <miopengemm/accuracytests.hpp>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/floattostring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/tinytwo.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFl>
void TinyOne<TFl>::initialise_cpu_mem(const TFl* a_, const TFl* b_, const TFl* c_)
{
  cpu_mem[Mat::E::A] = a_;
  cpu_mem[Mat::E::B] = b_;
  cpu_mem[Mat::E::C] = c_;
}

template <typename TFl>
void TinyOne<TFl>::initialise_common()
{

  rw_perms[Mat::E::A] = CL_MEM_READ_ONLY;
  rw_perms[Mat::E::B] = CL_MEM_READ_ONLY;
  rw_perms[Mat::E::C] = CL_MEM_READ_WRITE;

  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    mem_size[emat] = get_mat_memsize(gg, toff, emat);
  }

  gg.check_ldx_consistent();

  c_copy.resize(mem_size[Mat::E::C] / sizeof(TFl));
  std::memcpy(c_copy.data(), cpu_mem[Mat::E::C], mem_size[Mat::E::C]);

  opencl_memory_initialise();

  up_jinx.reset(new TinyZero(tgcq.command_queue,
                             gg,
                             toff,
                             gpu_safemem[Mat::E::A].clmem,
                             gpu_safemem[Mat::E::B].clmem,
                             gpu_safemem[Mat::E::C].clmem,
                             false,  // is not const
                             gpu_safemem_www.clmem,
                             mowri));
}

template <typename TFl>
TinyOne<TFl>::TinyOne(
  Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint, long)
  : gg(gg_),
    toff(toff_),
    cpu_mem(Mat::E::N),
    mowri(mowri_),
    tgcq(mowri,
         CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
         xhint,
         "command queue of TinyOne"),
    gpu_safemem(Mat::E::N, std::string("gpu_safemem vector of TinyOne")),
    gpu_safemem_www(std::string("gpu_safemem vector of TinyOne (www)")),
    mem_size(Mat::E::N),
    rw_perms(Mat::E::N)
{

  if (gg.derived.float_size_bytes != sizeof(TFl))
  {
    std::stringstream errm;
    errm << "float sizes don't agree in TinyOne. ";
    errm << "the size from geometry is " << gg.derived.float_size_bytes << ". ";
    errm << "the size from the template parameter is " << sizeof(TFl) << ".";
    throw miog_error(errm.str());
  }
}

template <typename TFl>
TinyOne<TFl>::TinyOne(Geometry        gg_,
                      Offsets         toff_,
                      const TFl*      a_,
                      const TFl*      b_,
                      const TFl*      c_,
                      owrite::Writer& mowri_,
                      const CLHint&   xhint)
  : TinyOne(gg_, toff_, mowri_, xhint, 42)

{
  initialise_cpu_mem(a_, b_, c_);
  initialise_common();
}

template <typename TFl>
TinyOne<TFl>::TinyOne(Geometry gg_,
                      Offsets  toff_,
                      std::array<const TFl*, Mat::E::N> abc_,
                      owrite::Writer& mowri_,
                      const CLHint&   xhint)
  : TinyOne(gg_, toff_, abc_[Mat::E::A], abc_[Mat::E::B], abc_[Mat::E::C], mowri_, xhint)
{
}

template <typename TFl>
void TinyOne<TFl>::initialise_cpu_mem_from_scratch()
{

  setabcw::set_abc<TFl>(
    {&__cpu_mem[Mat::E::A], &__cpu_mem[Mat::E::B], &__cpu_mem[Mat::E::C]}, gg, toff);

  //// why ?
  // for (auto& x : __cpu_mem[Mat::E::B])
  //{
  // x *= 1000;
  //}
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    cpu_mem[emat] = __cpu_mem[emat].data();
  }
}

template <typename TFl>
TinyOne<TFl>::TinyOne(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint)
  : TinyOne(gg_, toff_, mowri_, xhint, 42)

{
  initialise_cpu_mem_from_scratch();
  initialise_common();
}

template <typename TFl>
size_t TinyOne<TFl>::get_workspace_memsize()
{
  return get_total_workspace(gg, toff) * sizeof(TFl);
}

template <typename TFl>
void TinyOne<TFl>::opencl_memory_initialise()
{

  if (get_workspace_memsize() > 0)
  {

    std::stringstream hash;

    hash << "GPU Mem W (TinyOne) "
         << "with memory size " << get_workspace_memsize() << ".";

    oclutil::cl_set_buffer_from_command_queue(gpu_safemem_www.clmem,
                                              tgcq.command_queue,
                                              CL_MEM_READ_WRITE,
                                              get_workspace_memsize(),
                                              NULL,
                                              hash.str(),
                                              true);
  }

  // allocate memory for a,b,c on device, send it over
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    std::stringstream hash;

    hash << "GPU Mem " << Mat::M().name[emat] << " (TinyOne) "
         << "with memory size " << mem_size[emat] << ".";

    oclutil::cl_set_buffer_from_command_queue(gpu_safemem[emat].clmem,
                                              tgcq.command_queue,
                                              rw_perms[emat],
                                              mem_size[emat],
                                              NULL,
                                              hash.str(),
                                              true);

    oclutil::cl_enqueue_write_buffer(tgcq.command_queue,
                                     gpu_safemem[emat].clmem,
                                     CL_TRUE,
                                     0,
                                     mem_size[emat],
                                     cpu_mem[emat],
                                     0,
                                     NULL,
                                     NULL,
                                     std::string("enqueueing ") + Mat::M().name[emat] +
                                       " writebuff ",
                                     true);
  }
}

template <typename TFl>
std::vector<std::vector<double>> TinyOne<TFl>::benchgemm(const std::vector<HyPas>& hps,
                                                         const Halt&               hl)

{
  std::vector<std::vector<double>> times_s;
  for (auto& hp : hps)
  {
    times_s.push_back(up_jinx->benchgemm(hp, hl));
  }
  return times_s;
}

template <typename TFl>
Solution TinyOne<TFl>::find1(const FindParams& find_params, const Constraints& constraints)
{
  Solution tgs = up_jinx->find0(constraints, find_params);
  return tgs;
}

template <typename TFl>
void TinyOne<TFl>::accuracy_test(const HyPas& hp)  //, const TFl* c_true_for_test)
{

  // copy the const cpu matrix to the gpu
  oclutil::SafeClEvent event_write_c_to_gpu("accuracy test write");

  oclutil::cl_enqueue_write_buffer(tgcq.command_queue,
                                   gpu_safemem[Mat::E::C].clmem,
                                   CL_TRUE,
                                   0,
                                   mem_size[Mat::E::C],
                                   cpu_mem[Mat::E::C],
                                   0,
                                   nullptr,
                                   &event_write_c_to_gpu.clevent,
                                   "write of correct c in accuracy",
                                   true);

  // make sure the copy to gpu is complete
  oclutil::cl_wait_for_events(
    1, &event_write_c_to_gpu.clevent, "in accuracy test, waiting GEMM gpu ", true);

  // run gemm once on the gpu
  benchgemm({hp}, {{{0, 1}}, {{0, 1e12}}});

  // read the result to c_copy on the cpu
  oclutil::SafeClEvent event_read_c_back("accuracy test read");
  oclutil::cl_enqueue_read_buffer(tgcq.command_queue,
                                  gpu_safemem[Mat::E::C].clmem,
                                  CL_TRUE,
                                  0,
                                  mem_size[Mat::E::C],
                                  c_copy.data(),
                                  0,
                                  NULL,
                                  &event_read_c_back.clevent,
                                  "enqueue read to c, in base_basegemm_with_accuracy_test",
                                  true);

  // compute product on CPU.
  std::vector<TFl> c_for_cpu_compute(mem_size[Mat::E::C] / sizeof(TFl));
  std::memcpy(c_for_cpu_compute.data(), cpu_mem[Mat::E::C], mem_size[Mat::E::C]);

  cpugemm::gemm<TFl>(gg,
                     toff,
                     cpu_mem[Mat::E::A],
                     cpu_mem[Mat::E::B],
                     c_for_cpu_compute.data(),
                     Floating::get_default_alpha(),
                     Floating::get_default_beta(),
                     mowri);

  auto c_true_for_test = c_for_cpu_compute.data();

  // compute absolute GEMM : C <- alpha abs(A)abs(B) + beta abs(C).
  std::vector<TFl> A_abs(mem_size[Mat::E::A] / sizeof(TFl));
  std::memcpy(A_abs.data(), cpu_mem[Mat::E::A], mem_size[Mat::E::A]);
  for (auto& x : A_abs)
  {
    x = std::abs(x);
  }

  std::vector<TFl> B_abs(mem_size[Mat::E::B] / sizeof(TFl));
  std::memcpy(B_abs.data(), cpu_mem[Mat::E::B], mem_size[Mat::E::B]);
  for (auto& x : B_abs)
  {
    x = std::abs(x);
  }

  std::vector<TFl> C_abs(mem_size[Mat::E::C] / sizeof(TFl));
  std::memcpy(C_abs.data(), cpu_mem[Mat::E::C], mem_size[Mat::E::C]);
  for (auto& x : C_abs)
  {
    x = std::abs(x);
  }

  cpugemm::gemm<TFl>(gg,
                     toff,
                     A_abs.data(),
                     B_abs.data(),
                     C_abs.data(),
                     std::abs(Floating::get_default_alpha()),
                     std::abs(Floating::get_default_beta()),
                     mowri);

  // make sure the read back is complete complete
  oclutil::cl_wait_for_events(
    1, &event_read_c_back.clevent, "in accuracy test, waiting GEMM gpu ", true);

  std::stringstream errmss;
  errmss << "accuracy test in TinyOne,\n" << gg.get_string() << '\n' << hp.get_string() << '\n';
  // compare cpu and gpu results
  accuracytests::elementwise_compare(gg,
                                     toff,
                                     cpu_mem[Mat::E::C],  // before,
                                     c_true_for_test,     // after cpu
                                     c_copy.data(),       // after gpu
                                     C_abs.data(),        // after abs on cpu
                                     errmss.str(),
                                     mowri);
}

template class TinyOne<float>;
template class TinyOne<double>;
}
}
