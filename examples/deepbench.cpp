// Bencharking MIOpenGEMM on the DeepBench problem suite.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

//#ifdef __APPLE__
//#include <opencl.h>
//#else
#include <CL/cl.h>
//#endif

#include <miopengemm/gemm.hpp>

int main()
{

  // DeepBench problems, copied (12/09/2017) from DeepBench/code/kernels/gemm_problems.h
  // Vector saves m, n, k, a_t, b_t
  std::vector<std::tuple<int, int, int, bool, bool>> training_set = {
    std::make_tuple(1760, 16, 1760, false, false),
    std::make_tuple(1760, 32, 1760, false, false),
    std::make_tuple(1760, 64, 1760, false, false),
    std::make_tuple(1760, 128, 1760, false, false),
    std::make_tuple(1760, 7000, 1760, false, false),
    std::make_tuple(2048, 16, 2048, false, false),
    std::make_tuple(2048, 32, 2048, false, false),
    std::make_tuple(2048, 64, 2048, false, false),
    std::make_tuple(2048, 128, 2048, false, false),
    std::make_tuple(2048, 7000, 2048, false, false),
    std::make_tuple(2560, 16, 2560, false, false),
    std::make_tuple(2560, 32, 2560, false, false),
    std::make_tuple(2560, 64, 2560, false, false),
    std::make_tuple(2560, 128, 2560, false, false),
    std::make_tuple(2560, 7000, 2560, false, false),
    std::make_tuple(4096, 16, 4096, false, false),
    std::make_tuple(4096, 32, 4096, false, false),
    std::make_tuple(4096, 64, 4096, false, false),
    std::make_tuple(4096, 128, 4096, false, false),
    std::make_tuple(4096, 7000, 4096, false, false),
    std::make_tuple(1760, 16, 1760, true, false),
    std::make_tuple(1760, 32, 1760, true, false),
    std::make_tuple(1760, 64, 1760, true, false),
    std::make_tuple(1760, 128, 1760, true, false),
    std::make_tuple(1760, 7000, 1760, true, false),
    std::make_tuple(2048, 16, 2048, true, false),
    std::make_tuple(2048, 32, 2048, true, false),
    std::make_tuple(2048, 64, 2048, true, false),
    std::make_tuple(2048, 128, 2048, true, false),
    std::make_tuple(2048, 7000, 2048, true, false),
    std::make_tuple(2560, 16, 2560, true, false),
    std::make_tuple(2560, 32, 2560, true, false),
    std::make_tuple(2560, 64, 2560, true, false),
    std::make_tuple(2560, 128, 2560, true, false),
    std::make_tuple(2560, 7000, 2560, true, false),
    std::make_tuple(4096, 16, 4096, true, false),
    std::make_tuple(4096, 32, 4096, true, false),
    std::make_tuple(4096, 64, 4096, true, false),
    std::make_tuple(4096, 128, 4096, true, false),
    std::make_tuple(4096, 7000, 4096, true, false),
    std::make_tuple(1760, 7133, 1760, false, true),
    std::make_tuple(2048, 7133, 2048, false, true),
    std::make_tuple(2560, 7133, 2560, false, true),
    std::make_tuple(4096, 7133, 4096, false, true),
    std::make_tuple(5124, 9124, 1760, false, false),
    std::make_tuple(35, 8457, 1760, false, false),
    std::make_tuple(5124, 9124, 2048, false, false),
    std::make_tuple(35, 8457, 2048, false, false),
    std::make_tuple(5124, 9124, 2560, false, false),
    std::make_tuple(35, 8457, 2560, false, false),
    std::make_tuple(5124, 9124, 4096, false, false),
    std::make_tuple(35, 8457, 4096, false, false),
    std::make_tuple(5124, 9124, 1760, true, false),
    std::make_tuple(35, 8457, 1760, true, false),
    std::make_tuple(5124, 9124, 2048, true, false),
    std::make_tuple(35, 8457, 2048, true, false),
    std::make_tuple(5124, 9124, 2560, true, false),
    std::make_tuple(35, 8457, 2560, true, false),
    std::make_tuple(5124, 9124, 4096, true, false),
    std::make_tuple(35, 8457, 4096, true, false),
    std::make_tuple(7680, 16, 2560, false, false),
    std::make_tuple(7680, 32, 2560, false, false),
    std::make_tuple(7680, 64, 2560, false, false),
    std::make_tuple(7680, 128, 2560, false, false),
    std::make_tuple(7680, 16, 2560, true, false),
    std::make_tuple(7680, 32, 2560, true, false),
    std::make_tuple(7680, 64, 2560, true, false),
    std::make_tuple(7680, 128, 2560, true, false),
    std::make_tuple(3072, 16, 1024, false, false),
    std::make_tuple(3072, 32, 1024, false, false),
    std::make_tuple(3072, 64, 1024, false, false),
    std::make_tuple(3072, 128, 1024, false, false),
    std::make_tuple(3072, 16, 1024, true, false),
    std::make_tuple(3072, 32, 1024, true, false),
    std::make_tuple(3072, 64, 1024, true, false),
    std::make_tuple(3072, 128, 1024, true, false),
    std::make_tuple(3072, 7435, 1024, false, true),
    std::make_tuple(7680, 5481, 2560, false, true),
    std::make_tuple(512, 8, 500000, false, false),
    std::make_tuple(1024, 8, 500000, false, false),
    std::make_tuple(512, 16, 500000, false, false),
    std::make_tuple(1024, 16, 500000, false, false),
    std::make_tuple(512, 8, 500000, true, false),
    std::make_tuple(1024, 8, 500000, true, false),
    std::make_tuple(512, 16, 500000, true, false),
    std::make_tuple(1024, 16, 500000, true, false),
    std::make_tuple(1024, 700, 512, false, false),
    std::make_tuple(1024, 700, 512, true, false),
    std::make_tuple(7680, 24000, 2560, false, false),
    std::make_tuple(6144, 24000, 2560, false, false),
    std::make_tuple(4608, 24000, 1536, false, false),
    std::make_tuple(8448, 24000, 2816, false, false),
    std::make_tuple(3072, 24000, 1024, false, false),
    std::make_tuple(7680, 48000, 2560, false, false),
    std::make_tuple(6144, 48000, 2560, false, false),
    std::make_tuple(4608, 48000, 1536, false, false),
    std::make_tuple(8448, 48000, 2816, false, false),
    std::make_tuple(3072, 48000, 1024, false, false),
    std::make_tuple(7680, 24000, 2560, true, false),
    std::make_tuple(6144, 24000, 2560, true, false),
    std::make_tuple(4608, 24000, 1536, true, false),
    std::make_tuple(8448, 24000, 2816, true, false),
    std::make_tuple(3072, 24000, 1024, true, false),
    std::make_tuple(7680, 48000, 2560, true, false),
    std::make_tuple(6144, 48000, 2560, true, false),
    std::make_tuple(4608, 48000, 1536, true, false),
    std::make_tuple(8448, 48000, 2816, true, false),
    std::make_tuple(3072, 48000, 1024, true, false),
    std::make_tuple(6144, 16, 2560, false, false),
    std::make_tuple(4608, 16, 1536, false, false),
    std::make_tuple(8448, 16, 2816, false, false),
    std::make_tuple(6144, 32, 2560, false, false),
    std::make_tuple(4608, 32, 1536, false, false),
    std::make_tuple(8448, 32, 2816, false, false),
    std::make_tuple(6144, 16, 2560, true, false),
    std::make_tuple(4608, 16, 1536, true, false),
    std::make_tuple(8448, 16, 2816, true, false),
    std::make_tuple(6144, 32, 2560, true, false),
    std::make_tuple(4608, 32, 1536, true, false),
    std::make_tuple(8448, 32, 2816, true, false),
    std::make_tuple(512, 24000, 2816, false, false),
    std::make_tuple(512, 24000, 2048, false, false),
    std::make_tuple(512, 24000, 2560, false, false),
    std::make_tuple(512, 24000, 1536, false, false),
    std::make_tuple(1024, 24000, 2816, false, false),
    std::make_tuple(1024, 24000, 2048, false, false),
    std::make_tuple(1024, 24000, 2560, false, false),
    std::make_tuple(1024, 24000, 1536, false, false),
    std::make_tuple(512, 16, 512, false, false),
    std::make_tuple(1024, 16, 512, false, false),
    std::make_tuple(512, 24000, 2816, true, false),
    std::make_tuple(512, 24000, 2048, true, false),
    std::make_tuple(512, 24000, 2560, true, false),
    std::make_tuple(512, 24000, 1536, true, false),
    std::make_tuple(1024, 24000, 2816, true, false),
    std::make_tuple(1024, 24000, 2048, true, false),
    std::make_tuple(1024, 24000, 2560, true, false),
    std::make_tuple(1024, 24000, 1536, true, false),
    std::make_tuple(512, 16, 512, false, true),
    std::make_tuple(1024, 16, 512, false, true),
    std::make_tuple(512, 48000, 2816, false, false),
    std::make_tuple(512, 48000, 2048, false, false),
    std::make_tuple(512, 48000, 2560, false, false),
    std::make_tuple(512, 48000, 1536, false, false),
    std::make_tuple(1024, 48000, 2816, false, false),
    std::make_tuple(1024, 48000, 2048, false, false),
    std::make_tuple(1024, 48000, 2560, false, false),
    std::make_tuple(1024, 48000, 1536, false, false),
    std::make_tuple(512, 32, 512, false, false),
    std::make_tuple(1024, 32, 512, false, false),
    std::make_tuple(512, 48000, 2816, true, false),
    std::make_tuple(512, 48000, 2048, true, false),
    std::make_tuple(512, 48000, 2560, true, false),
    std::make_tuple(512, 48000, 1536, true, false),
    std::make_tuple(1024, 48000, 2816, true, false),
    std::make_tuple(1024, 48000, 2048, true, false),
    std::make_tuple(1024, 48000, 2560, true, false),
    std::make_tuple(1024, 48000, 1536, true, false),
    std::make_tuple(512, 32, 512, false, true),
    std::make_tuple(1024, 32, 512, false, true)};

  auto confirm = [](cl_int clstat) {
    if (clstat != CL_SUCCESS)
    {
      std::stringstream ss;
      ss << "OpenCL error status : " << clstat;
      throw std::runtime_error(ss.str());
    }
  };

  // (12/09/2017) as per line 59 of DeepBench/code/nvidia/gemm_bench.cu, alpha and beta are 1.
  const float alpha = 1.f;
  const float beta  = 1.f;

  // allocate host memories hA, hB, hC to be as large as the largest A, B and C in DeepBench.
  int sA = 0;
  int sB = 0;
  int sC = 0;
  for (auto& x : training_set)
  {
    sA = std::max<int>(sA, std::get<0>(x) * std::get<2>(x));
    sB = std::max<int>(sB, std::get<1>(x) * std::get<2>(x));
    sC = std::max<int>(sC, std::get<0>(x) * std::get<1>(x));
  }

  // 12/09/2017
  // as per DeepBench/code/nvidia/tensor.h, matrices are ~U[0,1].
  std::cout << "\nInitialise (largest) A, B, C from ~U[0,1] ... " << std::flush;

  std::vector<float> hA(sA);
  std::vector<float> hB(sB);
  std::vector<float> hC(sC);

  for (int i = 0; i < sA; ++i)
    hA[i]    = (rand() % 1000) / 1000.;
  for (int i = 0; i < sB; ++i)
    hB[i]    = (rand() % 1000) / 1000.;
  for (int i = 0; i < sC; ++i)
    hC[i]    = (rand() % 1000) / 1000.;

  std::cout << "done. \nInitialise OpenCL platform ... " << std::flush;
  cl_int  clstat;
  size_t  platform_id = 0;
  cl_uint num_platforms;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  std::vector<cl_platform_id> platforms(num_platforms);
  clstat = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  confirm(clstat);
  cl_platform_id platform = platforms[platform_id];

  size_t device_id = 0;
  std::cout << "done. \nInitialise OpenCL device #" << device_id << " ... " << std::flush;
  cl_uint num_devices;
  clstat = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
  std::vector<cl_device_id> devices(num_devices);
  clstat = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
  cl_device_id device = devices[device_id];

  std::cout << "done. \nInitialise OpenCL context ... " << std::flush;
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clstat);
  std::cout << "done. \nInitialise OpenCL queue ... " << std::flush;
  cl_queue_properties properties = 0;  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

#if (CL_VERSION_2_0 == 1)
  std::vector<cl_queue_properties> props = {CL_QUEUE_PROPERTIES, properties, 0};
  cl_command_queue Q = clCreateCommandQueueWithProperties(context, device, props.data(), &clstat);
#else
  cl_command_queue Q = clCreateCommandQueue(context, device, properties, &clstat);
#endif

  std::cout << "done. \nInitialise device memories ... " << std::flush;
  cl_mem dA = clCreateBuffer(context, CL_MEM_READ_WRITE, sA * sizeof(float), nullptr, &clstat);
  cl_mem dB = clCreateBuffer(context, CL_MEM_READ_WRITE, sB * sizeof(float), nullptr, &clstat);
  cl_mem dC = clCreateBuffer(context, CL_MEM_READ_WRITE, sC * sizeof(float), nullptr, &clstat);

  clEnqueueWriteBuffer(Q, dA, CL_TRUE, 0, sA * sizeof(float), hA.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(Q, dB, CL_TRUE, 0, sB * sizeof(float), hB.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(Q, dC, CL_TRUE, 0, sC * sizeof(float), hC.data(), 0, nullptr, nullptr);
  std::cout << "done. \nLaunch DeepBench.\n";

  std::cout << std::setw(30) << "Times" << std::endl;
  std::cout << std::setfill('-') << std::setw(95) << "-" << std::endl;
  std::cout << std::setfill(' ');
  int n_problems = training_set.size();
  std::cout
    << "    m       n      k      a_t     b_t   precision   time (usec)  tflops   numRepeats  (of "
    << n_problems << ")\n";

  bool cmaj = true;
  int  GID  = 1;
  for (auto& x : training_set)
  {
    size_t m   = std::get<0>(x);
    size_t n   = std::get<1>(x);
    size_t k   = std::get<2>(x);
    bool   tA  = std::get<3>(x);
    bool   tB  = std::get<4>(x);
    size_t lda = cmaj == tA ? k : m;
    size_t ldb = cmaj == tB ? n : k;
    size_t ldc = cmaj == false ? n : m;

    // as per line 95 DeepBench/code/nvidia/gemm_bench.cu, one call to warm-up.
    MIOpenGEMM::gemm0<float>(cmaj,  // column major
                             tA,
                             tB,
                             m,
                             n,
                             k,
                             alpha,
                             dA,
                             0,  // zero offset in A buffer
                             lda,
                             dB,
                             0,
                             ldb,
                             beta,
                             dC,
                             0,
                             ldc,
                             &Q,
                             0,
                             nullptr,
                             nullptr);

    // equivalent to cudaDeviceSynchronize() at line 127 of DeepBench/code/nvidia/gemm_bench.cu
    clFinish(Q);

    // as per line 70 of DeepBench/code/nvidia/gemm_bench.cu
    int  numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);
    auto start      = std::chrono::steady_clock::now();

    std::cout << std::setw(7) << m << std::setw(7) << n << std::setw(7) << k << std::setw(7)
              << (tA ? 't' : 'n') << std::setw(7) << (tB ? 't' : 'n') << std::setw(10) << "f32"
              << std::flush;

    for (int r = 0; r < numRepeats; ++r)
    {
      MIOpenGEMM::gemm0<float>(cmaj,
                               tA,
                               tB,
                               m,
                               n,
                               k,
                               alpha,
                               dA,
                               0,
                               lda,
                               dB,
                               0,
                               ldb,
                               beta,
                               dC,
                               0,
                               ldc,
                               &Q,
                               0,
                               nullptr,
                               nullptr);
    }
    // equivalent to cudaDeviceSynchronize() at line 164 of DeepBench/code/nvidia/gemm_bench.cu
    clFinish(Q);

    auto end = std::chrono::steady_clock::now();
    auto elapsed =
      static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);
    double tflops = (2. * m * n * k) / (1e6 * elapsed);

    std::cout << std::setw(14) << std::setprecision(4) << elapsed << std::setw(12) << tflops
              << std::setw(10) << numRepeats << std::setw(9) << GID << std::endl;
    ++GID;
  }
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseCommandQueue(Q);
  clReleaseContext(context);
}
