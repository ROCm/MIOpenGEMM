/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <miopengemm/enums.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>

#include <algorithm>
#include <vector>

// learing how to use float2 float4
int main()
{

  using namespace MIOpenGEMM;
  owrite::Writer                 mowri(Ver::E::TERMINAL, "");
  CLHint                         devhint;
  oclutil::CommandQueueInContext scq(mowri,
                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                       CL_QUEUE_PROFILING_ENABLE,
                                     devhint,
                                     "floatxdemo");

  size_t             ndata(1024);
  std::vector<float> v_data1(ndata);
  std::iota(v_data1.begin(), v_data1.end(), 0);

  std::vector<float> v_data2(v_data1);
  for (size_t i = 0; i < ndata; ++i)
  {
    v_data2[i] *= (1 - 2 * (i % 2 == 0));
  }

  std::vector<float> a_copy(ndata);

  std::array<float*, 2> cpu_mem{v_data1.data(), v_data2.data()};
  std::vector<oclutil::SafeClMem> gpu_safemem(2, std::string("safemem"));

  for (size_t i = 0; i < 2; ++i)
  {

    oclutil::cl_set_buffer_from_command_queue(gpu_safemem[i].clmem,
                                              scq.command_queue,
                                              CL_MEM_READ_WRITE,
                                              sizeof(float) * ndata,
                                              NULL,
                                              "set buffer floatxdemo",
                                              true);

    oclutil::cl_enqueue_write_buffer(scq.command_queue,
                                     gpu_safemem[i].clmem,
                                     CL_TRUE,
                                     0,
                                     sizeof(float) * ndata,
                                     cpu_mem[i],
                                     0,
                                     NULL,
                                     NULL,
                                     "enqueueing  writebuff ",
                                     true);
  }

  size_t gws{ndata / 4};
  size_t lws{64};

  cl_program clprog;
  cl_kernel  clkern;

  std::string kernstr = R"(
  
  
__attribute__((reqd_work_group_size(64,1, 1)))
__kernel void fadd
( 
__global float * restrict a, 
__global float * restrict b
)

{
  float2 * a2 = (float2 *) a; 
  float2 * b2 = (float2 *) b;
  
  const size_t local_id = get_local_id(0);
  const size_t group_id = get_group_id(0);
  
  size_t index = 2*(group_id*64 + local_id);
  a2 += index;
  b2 += index;
  for (size_t i = 0; i < 2; ++i){
    //a2[i].s0 += b2[i].s0;
    //a2[i].s1 += b2[i].s1;
    a2[i] *= b2[i];
  }
}
  
  
)";

  std::string fname = "fadd";

  cl_context   context;
  cl_device_id device_id;
  oclutil::cl_set_context_and_device_from_command_queue(
    scq.command_queue, context, device_id, mowri, true);

  oclutil::cl_set_program_and_kernel(
    context, device_id, kernstr, fname, clprog, clkern, "-cl-std=CL2.0 -Werror", mowri, true);

  for (size_t i = 0; i < 2; ++i)
  {
    oclutil::cl_set_kernel_arg(clkern,
                               i,
                               sizeof(cl_mem),
                               &gpu_safemem[i].clmem,
                               std::string("arg ") + std::to_string(i),
                               true);
  }

  oclutil::cl_flush(scq.command_queue, "flush", true);

  oclutil::cl_enqueue_ndrange_kernel(
    scq.command_queue, clkern, 1, NULL, &gws, &lws, 0, nullptr, nullptr, "enqueue", true);

  oclutil::cl_flush(scq.command_queue, "flush", true);

  oclutil::cl_enqueue_read_buffer(scq.command_queue,
                                  gpu_safemem[0].clmem,
                                  CL_TRUE,
                                  0,
                                  ndata * sizeof(float),
                                  a_copy.data(),
                                  0,
                                  NULL,
                                  nullptr,
                                  "enqueue read",
                                  true);

  oclutil::cl_flush(scq.command_queue, "flush", true);

  for (size_t i = 0; i < 20; ++i)
  {
    std::cout << v_data1[i] << " : " << a_copy[i] << std::endl;
  }
}
