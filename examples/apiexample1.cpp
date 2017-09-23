/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <iostream>
#include <limits>
#include <vector>
#include <miopengemm/gemm.hpp>

int main()
{

  bool isColMajor = false;
  bool tA         = false;
  bool tB         = false;

  size_t m = 2;
  size_t n = 2;
  size_t k = 3;

  size_t lda = 3;
  size_t ldb = 2;
  size_t ldc = 2;

  float alpha = 1.0;
  float beta  = 0.0;

  std::vector<float> A = {1, 1, 1, 2, 2, 2};
  std::vector<float> B = {2, 3, 2, 3, 2, 3};
  std::vector<float> C = {0, 1, std::numeric_limits<float>::quiet_NaN(), 3};
  std::vector<float> C_result(4);

  size_t platform_id = 0;
  size_t device_id   = 0;

  cl_uint num_platforms;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  cl_platform_id platform = platforms[platform_id];

  std::cout << "\nInitialise OpenCL device ... " << std::flush;
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
  std::vector<cl_device_id> devices(num_devices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
  cl_device_id device = devices[device_id];

  std::cout << "done. \nInitialise OpenCL context ... " << std::flush;
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

#if (CL_VERSION_2_0 == 1)
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
#else
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
#endif

  cl_event event = nullptr;

  std::cout << "done. \nInitialise device memories ... " << std::flush;
  cl_mem dev_a = clCreateBuffer(context, CL_MEM_READ_WRITE, 6 * sizeof(float), nullptr, nullptr);
  cl_mem dev_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 6 * sizeof(float), nullptr, nullptr);
  cl_mem dev_c = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(float), nullptr, nullptr);
  clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, 6 * sizeof(float), A.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, 6 * sizeof(float), B.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, dev_c, CL_TRUE, 0, 4 * sizeof(float), C.data(), 0, nullptr, nullptr);

  std::cout << "done. \nRun xgemm..." << std::flush;
  MIOpenGEMM::xgemm<float>(isColMajor,
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

  std::cout << "`done'. \nRead back to host..." << std::flush;
  clEnqueueReadBuffer(queue,
                      dev_c,
                      true,               // blocking read
                      0,                  // offset
                      4 * sizeof(float),  // cb
                      C_result.data(),    // read results to C_results
                      1,                  // wait for 1 event to complete :
                      &event,             // waiting for xgemm to complete before reading,
                      nullptr);

  std::cout << "done. \nPrint summary.\n" << std::flush;
  std::cout << "\n---> A \n"
            << A[0] << ' ' << A[1] << ' ' << A[2] << '\n'
            << A[3] << ' ' << A[4] << ' ' << A[5] << std::endl;

  std::cout << "\n---> B \n"
            << B[0] << ' ' << B[1] << '\n'
            << B[2] << ' ' << B[3] << '\n'
            << B[4] << ' ' << B[5] << std::endl;

  std::cout << "\n---> C \n" << C[0] << ' ' << C[1] << '\n' << C[2] << ' ' << C[3] << std::endl;

  std::cout << "\n---> alpha A B + beta C \n"
            << C_result[0] << "  " << C_result[1] << '\n'
            << C_result[2] << ' ' << C_result[3] << std::endl;

  clReleaseEvent(event);
  clReleaseMemObject(dev_a);
  clReleaseMemObject(dev_b);
  clReleaseMemObject(dev_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}
