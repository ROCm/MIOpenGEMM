/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <miopengemm/gemm.hpp>

int main(int argc, char* argv[])
{

  std::string message("U : for  uniform real:  [-1, 1].\n"
                      "P : for positive real:  [ 0, 1] (like DeepBench, see tensor.h 07/09/2017).\n"
                      "C : for constant real:  1.\n"
                      "example : `apiexample P'"
                      );

  if (argc != 2)
  {
    throw std::runtime_error("ERROR : 1 argument required.\n" + message);
  }

  //set the GEMM geometry.  
  bool isColMajor = false;
  bool tA         = false;
  bool tB         = false;

  size_t m = 5100;
  size_t n = 5100;
  size_t k = 5100;

  size_t lda = isColMajor == tA ? k : m;
  size_t ldb = isColMajor == tB ? n : k;
  size_t ldc = isColMajor == false ? n : m;

  size_t otha = isColMajor != tA ? k : m;
  size_t othb = isColMajor != tB ? n : k;
  size_t othc = isColMajor != false ? n : m;

  float alpha = 0.5236;
  float beta  = 1.2342;

  std::vector<float> host_a(lda * otha);
  std::vector<float> host_b(ldb * othb);
  std::vector<float> host_c(ldc * othc);

  //set be coeffient of random number based on user's option.  
  float X2 = 0;
  switch (argv[1][0])
  {
  case 'U':
    std::cout << "will populate matrices with uniform random [-1,1]\n";
    X2 = 2;
    break;

  case 'P':
    std::cout << "will populate matrices with uniform random [0,1]\n";
    X2 = 1;
    break;

  case 'C':
    std::cout << "will populate matrices with 1\n";
    X2 = 0;
    break;

  default: throw std::runtime_error("ERROR : unrecognised argument.\n" + message);
  }

  std::cout << "Initialise host memories ... " << std::flush;
  for (size_t i = 0; i < m * k; ++i)
  {
    host_a[i] = 1. - X2 * ((rand() % 1000) / 1000.);
  }
  for (size_t i = 0; i < n * k; ++i)
  {
    host_b[i] = 1. - X2 * ((rand() % 1000) / 1000.);
  }
  for (size_t i = 0; i < m * n; ++i)
  {
    host_c[i] = 1. - X2 * ((rand() % 1000) / 1000.);
  }

  size_t platform_id = 0;
  size_t device_id   = 0;

  std::cout << "done. \nInitialise OpenCL platform ... " << std::flush;
  cl_uint num_platforms;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  cl_platform_id platform = platforms[platform_id];

  std::cout << "done. \nInitialise OpenCL device ... " << std::flush;
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
  std::vector<cl_device_id> devices(num_devices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
  cl_device_id device = devices[device_id];

  std::cout << "done. \nInitialise OpenCL context ... " << std::flush;
  cl_context       context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  cl_command_queue queue   = clCreateCommandQueue(context, device, 0, nullptr);
  cl_event         event   = nullptr;

  std::cout << "done. \nInitialise device memories ... " << std::flush;
  cl_mem dev_a =
    clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, nullptr);
  cl_mem dev_b =
    clCreateBuffer(context, CL_MEM_READ_WRITE, n * k * sizeof(float), nullptr, nullptr);
  cl_mem dev_c =
    clCreateBuffer(context, CL_MEM_READ_WRITE, m * n * sizeof(float), nullptr, nullptr);
  clEnqueueWriteBuffer(
    queue, dev_a, CL_TRUE, 0, m * k * sizeof(float), host_a.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(
    queue, dev_b, CL_TRUE, 0, n * k * sizeof(float), host_b.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(
    queue, dev_c, CL_TRUE, 0, m * n * sizeof(float), host_c.data(), 0, nullptr, nullptr);

  // number of times to run GEMM for, excluding the 1 warm-up run.
  int n_runs = 10;

  std::cout
    << "done. \nFirst call with this GEMM geometry (generating OpenCL string, compiling, etc) ... "
    << std::flush;
  // the first, warm-up, call to xgemm. notice that ID is -1.   
  auto status = MIOpenGEMM::xgemm<float>(isColMajor,
                                         tA,
                                         tB,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         dev_a,
                                         0,
                                         lda,
                                         dev_b,
                                         0,
                                         ldb,
                                         beta,
                                         dev_c,
                                         0,
                                         ldc,
                                         nullptr,
                                         0,
                                         0,
                                         &queue,
                                         0,
                                         nullptr,
                                         &event,
                                         -1);

  // wait for warm-up call to finish.
  clWaitForEvents(1, &event);


  auto t0 = std::chrono::high_resolution_clock::now();
  
  std::cout << "done. \n"
            << n_runs << " calls with this geometry, blocking on the final one ... " << std::flush;
  
  // TODO : remove ID.
  //enqueue n_runs - 1 calls without any cl_event. 
  for (int i = 0; i < n_runs - 1; ++i)
  {
    status = MIOpenGEMM::xgemm<float>(isColMajor,
                                      tA,
                                      tB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      dev_a,
                                      0,
                                      lda,
                                      dev_b,
                                      0,
                                      ldb,
                                      beta,
                                      dev_c,
                                      0,
                                      ldc,
                                      nullptr,
                                      0,
                                      0,
                                      &queue,
                                      0,
                                      nullptr,
                                      nullptr,
                                      // notice that ID is now positive, cached kernel will be used. 
                                      status.ID); 
  }

  // The final call, and a wait for the event associated to it.
  status = MIOpenGEMM::xgemm<float>(isColMajor,
                                    tA,
                                    tB,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    dev_a,
                                    0,
                                    lda,
                                    dev_b,
                                    0,
                                    ldb,
                                    beta,
                                    dev_c,
                                    0,
                                    ldc,
                                    nullptr,
                                    0,
                                    0,
                                    &queue,
                                    0,
                                    nullptr,
                                    &event,
                                    status.ID);

  clWaitForEvents(1, &event);

  std::chrono::duration<double> fp_ms   = std::chrono::high_resolution_clock::now() - t0;
  auto                          seconds = fp_ms.count();
  auto                          gflops  = n_runs * (m * n * k * 2.) / (1e9 * seconds);

  std::cout << "done in " << seconds << " seconds [" << gflops << " gflops]" << std::endl;

  clReleaseEvent(event);
  clReleaseMemObject(dev_a);
  clReleaseMemObject(dev_b);
  clReleaseMemObject(dev_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}
