/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <miopengemm/bundle.hpp>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace standalone
{

std::string make(const Geometry& gg, const HyPas& hp, owrite::Writer& mowri)
{

  Offsets toff = get_zero_offsets();

  Derivabilty dblt(hp, gg);
  if (dblt.is_derivable == false)
  {
    throw miog_error("Non-derivable in standalone::make : " + dblt.msg);
  }

  kerngen::Bundle bundle(hp, gg);  //, mowri);

  if (bundle.v_tgks.size() != 1)
  {
    throw miog_error("Currently standalone::make only supports 1 kernel.");
  }

  std::array<size_t, Mat::E::N> n_elms;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    n_elms[emat] = get_mat_size(gg, toff, emat);
  }

  // we obtain the correct value

  srand(1011);
  std::array<std::vector<float>, Mat::E::N> vals;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    vals[emat].resize(n_elms[emat]);
    for (auto& x : vals[emat])
    {
      x = 1.0f - 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }

  cpugemm::gemm<float>(gg,
                       toff,
                       vals[Mat::E::A].data(),
                       vals[Mat::E::B].data(),
                       vals[Mat::E::C].data(),
                       Floating::get_default_alpha(),
                       Floating::get_default_beta(),
                       mowri);

  float sum_final_cpu = std::accumulate(vals[Mat::E::C].begin(), vals[Mat::E::C].end(), 0.f);

  std::stringstream ss;
  ss << R"(/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

// this cpp file was generated with function standalone::make of MIOpenGEMM, for
)";

  ss << "// geometry : " << gg.get_string() << '\n';
  ss << "// hyperparams : " << hp.get_string() << '\n';

  ss << R"( 

#include <CL/cl.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// print when not CL_SUCCESS
void checkstatus(cl_int x, std::string where)
{
  if (x != CL_SUCCESS)
  {
    std::cout << "\n" << where << " : exit status " << x << '.' << std::endl;
  }
}

int main()
{ 
)";

  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    ss << "  size_t " << Mat::M().lcase_name[emat] << "_n_elms = " << n_elms[emat] << ';' << '\n';
  }

  ss << R"(

  std::vector <float> a_init(a_n_elms);
  std::vector <float> b_init(b_n_elms);
  std::vector <float> c_init(c_n_elms);
  std::vector <float> c_final(c_n_elms);


  srand(1011);
  for (auto & x : a_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  for (auto & x : b_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }  

 for (auto & x : c_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }    
  
)";

  ss << "\n  std::string source_str = R\"(\n\n\n";
  ss << bundle.v_tgks[0].kernstr << "\n\n\n)\";";

  ss << R"(


  cl_platform_id   platform_id   = nullptr;
  cl_device_id     device_id     = nullptr;
  cl_context       context       = nullptr;
  cl_command_queue command_queue = nullptr;

  cl_mem memobj_a = nullptr;
  cl_mem memobj_b = nullptr;
  cl_mem memobj_c = nullptr;
  
)";

  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    ss << "  size_t " << Mat::M().lcase_name[emat] << "_offset = " << toff.offsets[emat] << ';'
       << '\n';
  }

  ss << R"(
  
  cl_program program = nullptr;
  cl_kernel  kernel  = nullptr;
  
  cl_uint    ret_num_devices;
  cl_uint    ret_num_platforms;
  cl_int     ret;

  /* Get platform/device information */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  checkstatus(ret, "clGetPlatformIDs");

  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  checkstatus(ret, "clGetDeviceIDs");

  /* Create OpenCL Context */
  context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
  checkstatus(ret, "clCreateContext");

  /* Create Command Queue */
  std::cout << "clCreateCommandQueue.." << std::flush;
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  checkstatus(ret, "clCreateCommandQueue");
  
  )";

  ss << "  // create and write to buffers \n";
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    ss << "  memobj_" << Mat::M().lcase_name[emat]
       << " = clCreateBuffer(context, CL_MEM_READ_WRITE, " << Mat::M().lcase_name[emat]
       << "_n_elms  * sizeof(float), nullptr, &ret);\n";
    ss << "  checkstatus(ret, \"clCreateBuffer\");\n";
    ss << "  ret = clEnqueueWriteBuffer(\n";
    ss << "  command_queue, memobj_" << Mat::M().lcase_name[emat] << ", CL_TRUE, 0, "
       << Mat::M().lcase_name[emat] << "_n_elms * sizeof(float), " << Mat::M().lcase_name[emat]
       << "_init.data(), 0, nullptr, nullptr);\n";
    ss << "  checkstatus(ret, \"clEnqueueWriteBuffer\");\n\n";
  }

  ss << R"(

  /* Create Kernel program from the read in source */
  auto   source_c_str = source_str.c_str();
  size_t source_size  = source_str.size();
  
  std::cout << "clCreateProgramWithSource.." << std::flush;
  program             = clCreateProgramWithSource(
    context, 1, (const char**)&source_c_str, (const size_t*)&source_size, &ret);
  checkstatus(ret, "clCreateProgramWithSource");

  /* Build Kernel Program */
  std::cout << "clBuildProgram.." << std::flush;
  ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  checkstatus(ret, "clBuildProgram");

  /* Create OpenCL Kernel */ 
  )";
  ss << "  kernel = clCreateKernel(program, \"" << bundle.v_tgks[0].fname << "\", &ret);\n";
  ss << "  checkstatus(ret, \"clCreateKernel\");\n";
  ss << "\n\n  /* Set OpenCL kernel argument */\n";

  ss << R"(

  std::cout << "clSetKernelArg(s).." << std::flush;

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_a);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 1, sizeof(size_t), &a_offset);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_b);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 3, sizeof(size_t), &b_offset);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&memobj_c);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 5, sizeof(size_t), &c_offset);
  checkstatus(ret, "clSetKernelArg");
)";

  ss << "  float alpha = " << std::setprecision(20) << Floating::get_default_alpha() << ";";
  ss << R"(
  ret = clSetKernelArg(kernel, 6, sizeof(float), &alpha);
  checkstatus(ret, "clSetKernelArg");
)";

  ss << "  float beta = " << std::setprecision(20) << Floating::get_default_beta() << ";";
  ss << R"(
  ret = clSetKernelArg(kernel, 7, sizeof(float), &beta);
  checkstatus(ret, "clSetKernelArg");
)";

  ss << "  size_t local_work_size = " << bundle.v_tgks[0].local_work_size << ";\n";
  ss << "  size_t global_work_size = " << bundle.v_tgks[0].global_work_size << ";\n";

  ss << R"(
  
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "clEnqueueNDRangeKernel.." << std::flush;
  ret = clEnqueueNDRangeKernel(
    command_queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
  checkstatus(ret, "clEnqueueNDRangeKernel");

  std::cout << "clFinish.." << std::flush;
  ret = clFinish(command_queue);
  checkstatus(ret, "clFinish");
  

  std::cout << "clEnqueueReadBuffer.." << std::flush;
  ret = clEnqueueReadBuffer(command_queue,
                                   memobj_c,
                                   CL_TRUE,
                                   0,
                                   sizeof(float)*c_n_elms,
                                   c_final.data(),
                                   0,
                                   nullptr,
                                   nullptr);
  
 checkstatus(ret, "clEnqueueReadBuffer");

  std::cout << "clFinish.." << std::flush;
  ret = clFinish(command_queue);
  checkstatus(ret, "clFinish");
 

    
  
  std::cout << "done." << std::endl;
  
  float sum_final = std::accumulate(c_final.begin(), c_final.end(), 0.f);
  float sum_init  = std::accumulate(c_init.begin(), c_init.end(), 0.f);

)";

  ss << "  // (precomputed in standalone.cpp using 3-for loops (not OpenBLAS))\n ";
  ss << "  float sum_final_cpu = " << std::setprecision(20) << sum_final_cpu << ";\n";
  ss << "  float error = sum_final_cpu - sum_final;\n";

  ss << R"(
  
  std::cout << "sum of initial c = " << std::setprecision(20) << sum_init << std::endl;
  std::cout << "sum of final c  gpu = " << std::setprecision(20) << sum_final << std::endl;
  std::cout << "sum of final on cpu = " <<  )"
     << std::setprecision(20) << sum_final_cpu << R"(  << std::endl; 
  std::cout << "(cpu - gpu )/cpu = " <<  std::setprecision(10) <<  error << std::endl; 

   
  

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

  
  
  )";

  return ss.str();
}
}
}
