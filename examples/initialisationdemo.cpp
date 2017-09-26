/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

void checkstatus(cl_int x, std::string where)
{
  if (x != CL_SUCCESS)
  {
    std::cout << where << "  " << x << std::endl;
  }
}

// Is initialisation with ints or floats different ? This program answers that
int main(int argc, char* argv[])
{

  std::string errm("call should be either `initialisationdemo float' or `initialisationdemo int'");
  if (argc != 2)
  {
    throw std::runtime_error(errm);
  }

  std::string tau = argv[1];
  if (tau != "int" && tau != "float")
  {
    throw std::runtime_error(errm);
  }

  size_t MEM_SIZE = 1024 * 1024;

  std::vector<float> vmem(MEM_SIZE);
  if (tau == "int")
  {
    for (size_t i = 0; i < MEM_SIZE; ++i)
    {
      vmem[i] = rand() % 16;
    }
  }

  else
  {
    for (size_t i = 0; i < MEM_SIZE; ++i)
    {
      vmem[i] = 1.f - 2.f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }
  auto mem = vmem.data();

  /* GEMM with m = n = k = 1024, macro tile is 128 x 128 */
  std::string source_str = R"(

__attribute__((reqd_work_group_size(256,1, 1)))
__kernel void gemm_demo
(
__global const float * restrict a, 
__global const float * restrict b, 
__global float       *          c
)
{
  const size_t local_id = get_local_id(0);
  const size_t group_id_xy = get_group_id(0);
  const size_t micro_id_a = local_id % 16;
  const size_t micro_id_b = local_id / 16;
  size_t group_id_b = group_id_xy % 8;
  size_t group_id_a = group_id_xy / 8;
  __local float localA[2064];
  __local const float * lA;
  float rA[8];
  size_t write_macro_tile_start_a = group_id_a*128; 
  const size_t write_start_a = write_macro_tile_start_a + micro_id_a*8;
  const size_t pll_unroll_a_load_id = local_id % 16;
  const size_t perp_unroll_a_load_id = local_id / 16;
  size_t read_macro_tile_start_a = group_id_a*128; 
  a += read_macro_tile_start_a*1024;
  const size_t a_offset_pll_unroll = 1 * pll_unroll_a_load_id;
  const size_t a_offset_perp_unroll = 8 * perp_unroll_a_load_id;
  a += 1 * a_offset_pll_unroll;
  a += 1024 * a_offset_perp_unroll;
  __local float localB[2064];
  __local const float * lB;
  float rB[8];
  size_t write_macro_tile_start_b = group_id_b*128; 
  const size_t write_start_b = write_macro_tile_start_b + micro_id_b*8;
  const size_t pll_unroll_b_load_id = local_id % 16;
  const size_t perp_unroll_b_load_id = local_id / 16;
  size_t read_macro_tile_start_b = group_id_b*128; 
  b += read_macro_tile_start_b*1024;
  const size_t b_offset_pll_unroll = 1 * pll_unroll_b_load_id;
  const size_t b_offset_perp_unroll = 8 * perp_unroll_b_load_id;
  b += 1 * b_offset_pll_unroll;
  b += 1024 * b_offset_perp_unroll;
  float rC[8][8] = {{0.}};
  int n_unrolls_remaining = 1024 / 16;
  while (n_unrolls_remaining > 0){
    for (size_t mu_perp_i = 0; mu_perp_i < 8; ++mu_perp_i) {
      localA[129*(a_offset_pll_unroll) + (a_offset_perp_unroll + mu_perp_i)] = a[mu_perp_i*1024];
    }
    a += 1*16;
    for (size_t mu_perp_i = 0; mu_perp_i < 8; ++mu_perp_i) {
      localB[129*(b_offset_pll_unroll) + (b_offset_perp_unroll + mu_perp_i)] = b[mu_perp_i*1024];
    }
    b += 1*16;
    barrier(CLK_LOCAL_MEM_FENCE); 
    lA = localA + micro_id_a*8;
    lB = localB + micro_id_b*8;
    for (size_t u = 0; u < 16; ++u){
      for (size_t i = 0; i < 8; ++i){
        rA[i] = lA[i*1];
      }
      lA += 129;
      for (size_t i = 0; i < 8; ++i){
        rB[i] = lB[i*1];
      }
      lB += 129;
      for (size_t row = 0; row < 8; ++row){
        for (size_t col = 0; col < 8; ++col){
          rC[row][col] += rA[row]*rB[col];   
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
    --n_unrolls_remaining;
  }
  size_t index;
  for (size_t row = 0; row < 8; ++row) {
    for (size_t col = 0; col < 8; ++col) {
      index = (write_start_a + row) + 1024*(write_start_b + col);
      c[index] *= 0.8345345723452346235;
      c[index] += 1.3124234524523452342*rC[row][col];
    }
  }
}
  
  
)";

  cl_platform_id   platform_id   = NULL;
  cl_device_id     device_id     = NULL;
  cl_context       context       = NULL;
  cl_command_queue command_queue = NULL;

  cl_mem memobj_a = NULL;
  cl_mem memobj_b = NULL;
  cl_mem memobj_c = NULL;

  cl_program program = NULL;
  cl_kernel  kernel  = NULL;
  cl_uint    ret_num_devices;
  cl_uint    ret_num_platforms;
  cl_int     ret;

  /* Get platform/device information */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  checkstatus(ret, "clGetPlatformIDs");

  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  checkstatus(ret, "clGetDeviceIDs");

  /* Create OpenCL Context */
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  checkstatus(ret, "clCreateContext");

  /* Create Command Queue */

  // command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  cl_queue_properties properties =
    CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#if (CL_VERSION_2_0 == 1)
  std::vector<cl_queue_properties> props = {CL_QUEUE_PROPERTIES, properties, 0};
  command_queue = clCreateCommandQueueWithProperties(context, device_id, props.data(), nullptr);
#else
  command_queue = clCreateCommandQueue(context, device_id, properties, nullptr);
#endif

  checkstatus(ret, "clCreateCommandQueue");

  /* Create memory buffers */
  memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);
  checkstatus(ret, "clCreateBuffer");

  memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);
  checkstatus(ret, "clCreateBuffer");

  memobj_c = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);
  checkstatus(ret, "clCreateBuffer");

  /* Transfer data to memory buffer */
  ret = clEnqueueWriteBuffer(
    command_queue, memobj_a, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);
  checkstatus(ret, "clEnqueueWriteBuffer");

  ret = clEnqueueWriteBuffer(
    command_queue, memobj_b, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);
  checkstatus(ret, "clEnqueueWriteBuffer");

  ret = clEnqueueWriteBuffer(
    command_queue, memobj_c, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);
  checkstatus(ret, "clEnqueueWriteBuffer");

  /* Create Kernel program from the read in source */
  auto   source_c_str = source_str.c_str();
  size_t source_size  = source_str.size();
  program             = clCreateProgramWithSource(context,
                                      1,
                                      static_cast<const char**>(&source_c_str),
                                      static_cast<const size_t*>(&source_size),
                                      &ret);
  checkstatus(ret, "clCreateProgramWithSource");

  /* Build Kernel Program */
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  checkstatus(ret, "clBuildProgram");

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "gemm_demo", &ret);
  checkstatus(ret, "clCreateKernel");

  /* Set OpenCL kernel argument */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void*>(&memobj_a));
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void*>(&memobj_b));
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), static_cast<void*>(&memobj_c));
  checkstatus(ret, "clSetKernelArg");

  size_t local_work_size  = 256;
  size_t global_work_size = 256 * 64;

  /* Execute OpenCL kernel several times */
  auto start = std::chrono::high_resolution_clock::now();

  int n_iterations = 4000;
  std::cout << "kernel execution loop (" << n_iterations << " runs)..." << std::flush;
  ret = clEnqueueNDRangeKernel(
    command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
  checkstatus(ret, "clEnqueueNDRangeKernel");
  for (size_t i = 1; i < n_iterations; ++i)
  {
    ret = clEnqueueNDRangeKernel(
      command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
  }

  ret = clFinish(command_queue);
  checkstatus(ret, "clFinish");
  std::cout << "done." << std::endl;

  auto                         end             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms           = end - start;
  float                        elapsed_seconds = fp_ms.count();
  std::cout << "elapsed seconds : " << elapsed_seconds << std::endl;

  /* Finalization */
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj_a);
  ret = clReleaseMemObject(memobj_b);
  ret = clReleaseMemObject(memobj_c);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  return 0;
}
