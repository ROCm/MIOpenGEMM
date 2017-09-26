/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <miopengemm/gemm.hpp>

int driver(bool   isColMajor,
           bool   tA,
           bool   tB,
           size_t m,
           size_t n,
           size_t k,
           float  alpha,
           float  beta,
           bool   fast_ID,
           char   init,
           size_t n_runs)
{

  size_t lda = isColMajor == tA ? k : m;
  size_t ldb = isColMajor == tB ? n : k;
  size_t ldc = isColMajor == false ? n : m;

  size_t otha = isColMajor != tA ? k : m;
  size_t othb = isColMajor != tB ? n : k;
  size_t othc = isColMajor != false ? n : m;

  std::vector<float> host_a(lda * otha);
  std::vector<float> host_b(ldb * othb);
  std::vector<float> host_c(ldc * othc);

  // the coeffient of random number based on user's option.
  float X2 = 0;
  switch (init)  // argv[1][0])
  {
  case 'U':
    std::cout << "initialise hose matrices with uniform random [-1,1] ... ";
    X2 = 2;
    break;

  case 'P':
    std::cout << "initialise hose matrices with uniform random [0,1] ... ";
    X2 = 1;
    break;

  case 'C':
    std::cout << "initialise hose matrices matrices with 1.0 ... ";
    X2 = 0;
    break;

  default: throw std::runtime_error("ERROR : unrecognised -d flag.");
  }

  for (size_t i = 0; i < m * k; ++i)
  {
    host_a[i] = 1.0f - X2 * ((rand() % 1000) / 1000.0f);
  }
  for (size_t i = 0; i < n * k; ++i)
  {
    host_b[i] = 1.0f - X2 * ((rand() % 1000) / 1000.0f);
  }
  for (size_t i = 0; i < m * n; ++i)
  {
    host_c[i] = 1.0f - X2 * ((rand() % 1000) / 1000.0f);
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
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

  cl_queue_properties properties =
    CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

#if (CL_VERSION_2_0 == 1)
  std::vector<cl_queue_properties> props = {CL_QUEUE_PROPERTIES, properties, 0};
  cl_command_queue                 queue =
    clCreateCommandQueueWithProperties(context, device, props.data(), nullptr);
#else
  cl_command_queue queue = clCreateCommandQueue(context, device, properties, nullptr);
#endif

  cl_event event = nullptr;

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
  auto errstat = clWaitForEvents(1, &event);
  if (errstat != CL_SUCCESS)
  {
    std::cout << "Failed to wait in warm-up, error code : " << errstat << std::endl;
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  std::cout << "done. \n"
            << n_runs << " calls "
            << "(fast_ID=" << fast_ID << ")"
            << " blocking on the final one ... " << std::flush;

  // we have the correct ID from the warm-up call. Should it be used? (makes small problems slightly
  // faster)
  int loop_ID = fast_ID ? status.ID : -1;
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
                                      loop_ID);
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

  auto x = clWaitForEvents(1, &event);
  if (x != CL_SUCCESS)
  {
    std::cout << "Failed to wait for final, error code : " << x << std::endl;
  }

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

int main(int argc, char* argv[])
{

  std::map<std::string, size_t> values;
  values["m"]          = 5100;
  values["n"]          = 5100;
  values["k"]          = 5100;
  values["isColMajor"] = 0;
  values["tA"]         = 0;
  values["tB"]         = 0;
  values["id"]         = 1;
  int   n_runs         = -1;  // number of times to run GEMM, excluding the first run.
  char  init           = 'U';
  float alpha          = 1.0f;
  float beta           = 1.0f;

  std::stringstream helpss;
  helpss << "----------------------\nOptions:\n----------------------\n";
  helpss << "flag -id indicates whether to use ID directly after first call.\n";

  for (auto& x : values)
  {
    auto k = std::get<0>(x);
    auto v = std::get<1>(x);
    helpss << '-' << k << " [default = " << v << "]\n";
  }
  helpss << "-n_runs [default = min(1500, max(1e11 / (2kmn), 2))]\n";
  helpss << "-d [default = 'U' options are: \n"
         << "   U : for  uniform real:  [-1, 1].\n"
         << "   P : for positive real:  [ 0, 1] (like DeepBench, see tensor.h 07/09/2017).\n"
         << "   C : for constant real in all matrices:  1.]\n";

  helpss << "-alpha [default = 1.0]: \n";
  helpss << "-beta  [default = 1.0]: \n";

  std::string help = helpss.str();

  std::vector<std::string> frags;
  for (int i = 0; i < argc; ++i)
  {
    frags.push_back(argv[i]);
  }

  // check if help requested
  for (auto& f : frags)
  {
    if (f == "--help" || f == "-h" || f == "-help")
    {
      std::cout << help << std::endl;
      return 0;
    }
  }

  for (int i = 0; i < frags.size() - 1; ++i)
  {
    if (frags[i][0] != '-')
    {
    }
    else
    {
      if (frags[i] == "-n_runs")
      {
        n_runs = std::stoi(frags[i + 1]);
      }
      else if (frags[i] == "-d")
      {
        init = frags[i + 1][0];
      }

      else if (frags[i] == "-beta")
      {
        beta = std::stod(frags[i + 1]);
      }

      else if (frags[i] == "-alpha")
      {
        alpha = std::stod(frags[i + 1]);
      }

      else
      {
        auto key = frags[i].substr(1);
        if (values.count(key) != 0)
        {
          values[key] = std::stoi(frags[i + 1]);
        }
      }
    }
  }

  bool   isColMajor = static_cast<bool>(values["isColMajor"]);
  bool   tA         = static_cast<bool>(values["tA"]);
  bool   tB         = static_cast<bool>(values["tB"]);
  size_t m          = values["m"];
  size_t n          = values["n"];
  size_t k          = values["k"];
  bool   fast_ID    = values["id"];

  if (n_runs <= 0)
  {
    n_runs = std::min<size_t>(1500, std::max<size_t>(std::ceil(1e11 / (2 * m * k * n)), 2));
  }

  std::cout << "running with: \n";
  for (auto& x : values)
  {
    std::cout << "   " << std::get<0>(x) << " = " << std::get<1>(x) << '\n';
  }
  std::cout << "   n_runs = " << n_runs << '\n';
  std::cout << "   init = " << init << '\n' << '\n';

  int X = driver(isColMajor, tA, tB, m, n, k, alpha, beta, fast_ID, init, n_runs);
  return X;
}
