/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <sstream>
#include <vector>
#include <algorithm>
//#include <miopengemm/basicfind.hpp>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/geometryutil.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

int run_find_experiments(const std::vector<Geometry>& geometries,
                         std::vector<std::string>&    v_constraints,
                         const FindParams&            find_params,
                         bool                         verbose_inner,
                         std::string                  basedir_inner,
                         bool                         verbose_outer,
                         std::string                  fn_outer)
{

  owrite::Writer mowri_outer(Ver::E::TERMINAL, ""); // TODO : fix verbose_outer, fn_outer != "", fn_outer);

  std::string cache_write_string("");

  Offsets  offsets(13, 24, 35, 46, 57, 67, 79, 101);
  size_t n_postfind_runs = 0;
  bool     do_cpu_test     = false;

  // We're tracking the overall run time with these
  auto                         start           = std::chrono::high_resolution_clock::now();
  auto                         end             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms           = end - start;
  float                        elapsed_seconds = fp_ms.count();

  for (auto& constraints : v_constraints)
  {

    std::string fulldir_inner;
    if (basedir_inner != "")
    {
      if (basedir_inner.back() != '/')
      {
        basedir_inner += "/";
      }

      std::stringstream fulldir_inner_ss;
      fulldir_inner_ss << basedir_inner << "cs" << constraints << "/";
      fulldir_inner = fulldir_inner_ss.str();
      /* TODO WARNING : only works on Linux (makes directory)  */
      std::string syscall = std::string("mkdir -p  ") + fulldir_inner;
      int         v_sys   = std::system(syscall.c_str());

      if (v_sys != 0)
      {
        std::stringstream ss;
        ss << "The following call to std::system failed:\n"
           << syscall << "\nIs the operating system posix? If not, the call to "
           << "std::system should be "
           << "different.\nPlease raise an issue if you see this, "
           << "and ideally propose a fix";
        throw miog_error(ss.str());
      }
    }

    mowri_outer << "\nEntering experiment with constraints_string = `" << constraints << "'"
                << Endl;

    for (size_t prob_i = 0; prob_i < geometries.size(); ++prob_i)
    {
      Geometry          gg = geometries[prob_i];
      std::stringstream ss_logfile;
      if (basedir_inner != "")
      {
        ss_logfile << fulldir_inner << gg.get_string() << ".txt";
      }
      mowri_outer << (prob_i + 1) << "/" << geometries.size() << "       " << gg.get_string()
                  << Endl;

      std::string logfile = ss_logfile.str();

      if (basedir_inner != "")
      {
        mowri_outer << logfile << Endl;
      }

    
      dev::Boa goa(gg, offsets, mowri_outer); //TODO : should be a mowri
      Solution soln = goa.find(find_params, constraints);

      //auto soln = dev::basicfind(
      //find_params,
      //constraints, 
      //gg,
      //offsets,
                            
                            //mowri_outer // TODO : this is obviously not correct, fix it.
                            //);
                            
      end = std::chrono::high_resolution_clock::now();

      cache_write_string += soln.get_cache_entry_string();

      fp_ms           = end - start;
      elapsed_seconds = fp_ms.count();

      if (basedir_inner == "")
      {
        mowri_outer << " \t " << Endl;
      }

      mowri_outer << "soln median gflops :  " << soln.statistics.median_benchmark_gflops
                  << "  \t soln median time : " << soln.statistics.median_benchmark_time
                  << "  \t  elapsed time : " << elapsed_seconds << " [s] " << Endl;

      if (basedir_inner != "")
      {
        mowri_outer << "\n";
      }
    }
  }

  mowri_outer << "\nAll experiments have completed. To cache the best kernels "
                 "found, copy the "
                 "following string (between the `snips') into kernelcache.cpp"
              << Endl;
  mowri_outer << "\n\n-- snip -- snip -- snip -- \n\n\n";
  mowri_outer << cache_write_string << Flush;
  mowri_outer << "\n-- snip -- snip -- snip -- \n " << Endl;

  return 0;
}

std::vector<Geometry> get_old_deepbench_geometries(size_t workspace_size)
{

  std::vector<std::tuple<size_t, size_t, size_t, bool, bool>> baiduproblems = {
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
    // std::make_tuple(4096, 7133, 4096, false, true),
  };

  return get_from_m_n_k_tA_tB(baiduproblems, workspace_size);
}


std::vector<Geometry> get_deepbench_geometries(size_t workspace_size)
{

  std::vector<std::tuple<size_t, size_t, size_t, bool, bool>> baiduproblems = {
    std::make_tuple(512, 8, 500000, true, false),
    std::make_tuple(512, 8, 500000, false, false),
    std::make_tuple(512, 16, 512, false, true),
    std::make_tuple(512, 16, 512, false, false),
    std::make_tuple(512, 16, 500000, true, false),
    std::make_tuple(1024, 8, 500000, true, false),
    std::make_tuple(512, 16, 500000, false, false),
    std::make_tuple(1024, 8, 500000, false, false),
    std::make_tuple(512, 32, 512, false, true),
    std::make_tuple(512, 32, 512, false, false),
    std::make_tuple(1024, 16, 512, false, true),
    std::make_tuple(1024, 16, 512, false, false),
    std::make_tuple(1024, 16, 500000, true, false),
    std::make_tuple(1024, 16, 500000, false, false),
    std::make_tuple(1760, 16, 1760, true, false),
    std::make_tuple(1760, 16, 1760, false, false),
    std::make_tuple(1024, 32, 512, false, true),
    std::make_tuple(1024, 32, 512, false, false),
    std::make_tuple(2048, 16, 2048, true, false),
    std::make_tuple(2048, 16, 2048, false, false),
    std::make_tuple(2560, 16, 2560, true, false),
    std::make_tuple(2560, 16, 2560, false, false),
    std::make_tuple(3072, 16, 1024, true, false),
    std::make_tuple(3072, 16, 1024, false, false),
    std::make_tuple(1760, 32, 1760, true, false),
    std::make_tuple(1760, 32, 1760, false, false),
    std::make_tuple(4096, 16, 4096, true, false),
    std::make_tuple(2048, 32, 2048, true, false),
    std::make_tuple(4096, 16, 4096, false, false),
    std::make_tuple(2048, 32, 2048, false, false),
    std::make_tuple(4608, 16, 1536, true, false),
    std::make_tuple(4608, 16, 1536, false, false),
    std::make_tuple(2560, 32, 2560, true, false),
    std::make_tuple(2560, 32, 2560, false, false),
    std::make_tuple(6144, 16, 2048, true, false),
    std::make_tuple(6144, 16, 2048, false, false),
    std::make_tuple(3072, 32, 1024, true, false),
    std::make_tuple(3072, 32, 1024, false, false),
    std::make_tuple(1760, 64, 1760, true, false),
    std::make_tuple(1760, 64, 1760, false, false),
    std::make_tuple(7680, 16, 2560, true, false),
    std::make_tuple(7680, 16, 2560, false, false),
    std::make_tuple(4096, 32, 4096, true, false),
    std::make_tuple(2048, 64, 2048, true, false),
    std::make_tuple(4096, 32, 4096, false, false),
    std::make_tuple(2048, 64, 2048, false, false),
    std::make_tuple(8448, 16, 2816, true, false),
    std::make_tuple(8448, 16, 2816, false, false),
    std::make_tuple(4608, 32, 1536, true, false),
    std::make_tuple(4608, 32, 1536, false, false),
    std::make_tuple(2560, 64, 2560, true, false),
    std::make_tuple(2560, 64, 2560, false, false),
    std::make_tuple(6144, 32, 2048, true, false),
    std::make_tuple(6144, 32, 2048, false, false),
    std::make_tuple(3072, 64, 1024, true, false),
    std::make_tuple(3072, 64, 1024, false, false),
    std::make_tuple(1760, 128, 1760, true, false),
    std::make_tuple(1760, 128, 1760, false, false),
    std::make_tuple(7680, 32, 2560, true, false),
    std::make_tuple(7680, 32, 2560, false, false),
    std::make_tuple(4096, 64, 4096, true, false),
    std::make_tuple(2048, 128, 2048, true, false),
    std::make_tuple(4096, 64, 4096, false, false),
    std::make_tuple(2048, 128, 2048, false, false),
    std::make_tuple(8448, 32, 2816, true, false),
    std::make_tuple(8448, 32, 2816, false, false),
    std::make_tuple(35, 8457, 4096, true, false),
    std::make_tuple(35, 8457, 2560, true, false),
    std::make_tuple(35, 8457, 2048, true, false),
    std::make_tuple(35, 8457, 1760, true, false),
    std::make_tuple(35, 8457, 4096, false, false),
    std::make_tuple(35, 8457, 2560, false, false),
    std::make_tuple(35, 8457, 2048, false, false),
    std::make_tuple(35, 8457, 1760, false, false),
    std::make_tuple(2560, 128, 2560, true, false),
    std::make_tuple(2560, 128, 2560, false, false),
    std::make_tuple(3072, 128, 1024, true, false),
    std::make_tuple(3072, 128, 1024, false, false),
    std::make_tuple(7680, 64, 2560, true, false),
    std::make_tuple(7680, 64, 2560, false, false),
    std::make_tuple(4096, 128, 4096, true, false),
    std::make_tuple(4096, 128, 4096, false, false),
    std::make_tuple(1024, 700, 512, true, false),
    std::make_tuple(1024, 700, 512, false, false),
    std::make_tuple(7680, 128, 2560, true, false),
    std::make_tuple(7680, 128, 2560, false, false),
    std::make_tuple(512, 24000, 1530, true, false),
    std::make_tuple(512, 24000, 2560, true, false),
    std::make_tuple(512, 24000, 2048, true, false),
    std::make_tuple(512, 24000, 2816, true, false),
    std::make_tuple(512, 24000, 1530, false, false),
    std::make_tuple(512, 24000, 2560, false, false),
    std::make_tuple(512, 24000, 2048, false, false),
    std::make_tuple(512, 24000, 2816, false, false),
    std::make_tuple(1760, 7000, 1760, true, false),
    std::make_tuple(1760, 7000, 1760, false, false),
    std::make_tuple(1760, 7133, 1760, false, true),
    std::make_tuple(2048, 7000, 2048, true, false),
    std::make_tuple(2048, 7000, 2048, false, false),
    std::make_tuple(2048, 7133, 2048, false, true),
    std::make_tuple(2560, 7000, 2560, true, false),
    std::make_tuple(2560, 7000, 2560, false, false),
    std::make_tuple(2560, 7133, 2560, false, true),
    std::make_tuple(3072, 7435, 1024, false, true),
    std::make_tuple(512, 48000, 1530, true, false),
    std::make_tuple(512, 48000, 2560, true, false),
    std::make_tuple(512, 48000, 2048, true, false),
    std::make_tuple(512, 48000, 2816, true, false),
    std::make_tuple(512, 48000, 1530, false, false),
    std::make_tuple(512, 48000, 2560, false, false),
    std::make_tuple(512, 48000, 2048, false, false),
    std::make_tuple(512, 48000, 2816, false, false),
    std::make_tuple(1024, 24000, 1530, true, false),
    std::make_tuple(1024, 24000, 2560, true, false),
    std::make_tuple(1024, 24000, 2048, true, false),
    std::make_tuple(1024, 24000, 2816, true, false),
    std::make_tuple(1024, 24000, 1530, false, false),
    std::make_tuple(1024, 24000, 2560, false, false),
    std::make_tuple(1024, 24000, 2048, false, false),
    std::make_tuple(1024, 24000, 2816, false, false),
    std::make_tuple(4096, 7000, 4096, true, false),
    std::make_tuple(4096, 7000, 4096, false, false),
    std::make_tuple(4096, 7133, 4096, false, true),
    std::make_tuple(7680, 5481, 2560, false, true),
    std::make_tuple(5124, 9124, 4096, true, false),
    std::make_tuple(5124, 9124, 2560, true, false),
    std::make_tuple(5124, 9124, 2048, true, false),
    std::make_tuple(5124, 9124, 1760, true, false),
    std::make_tuple(5124, 9124, 4096, false, false),
    std::make_tuple(5124, 9124, 2560, false, false),
    std::make_tuple(5124, 9124, 2048, false, false),
    std::make_tuple(5124, 9124, 1760, false, false),
    std::make_tuple(1024, 48000, 1530, true, false),
    std::make_tuple(1024, 48000, 2560, true, false),
    std::make_tuple(1024, 48000, 2048, true, false),
    std::make_tuple(1024, 48000, 2816, true, false),
    std::make_tuple(1024, 48000, 1530, false, false),
    std::make_tuple(1024, 48000, 2560, false, false),
    std::make_tuple(1024, 48000, 2048, false, false),
    std::make_tuple(1024, 48000, 2816, false, false),
    std::make_tuple(3072, 24000, 1024, true, false),
    std::make_tuple(3072, 24000, 1024, false, false),
    std::make_tuple(4608, 24000, 1536, true, false),
    std::make_tuple(4608, 24000, 1536, false, false),
    std::make_tuple(3072, 48000, 1024, true, false),
    std::make_tuple(6144, 24000, 2048, true, false),
    std::make_tuple(3072, 48000, 1024, false, false),
    std::make_tuple(6144, 24000, 2048, false, false),
    std::make_tuple(7680, 24000, 2560, true, false),
    std::make_tuple(7680, 24000, 2560, false, false),
    std::make_tuple(8448, 24000, 2816, true, false),
    std::make_tuple(8448, 24000, 2816, false, false),
    std::make_tuple(4608, 48000, 1536, true, false),
    std::make_tuple(4608, 48000, 1536, false, false),
    std::make_tuple(6144, 48000, 2048, true, false),
    std::make_tuple(6144, 48000, 2048, false, false),
    std::make_tuple(7680, 48000, 2560, true, false),
    std::make_tuple(7680, 48000, 2560, false, false),
    std::make_tuple(8448, 48000, 2816, true, false),
    std::make_tuple(8448, 48000, 2816, false, false)
  };
  return get_from_m_n_k_tA_tB(baiduproblems, workspace_size);
}


std::vector<Geometry> get_new_deepbench_geometries(size_t workspace_size)
{
  auto allg = get_deepbench_geometries(workspace_size);
  std::vector<std::string> oldg_strings;
  for ( auto & x : get_old_deepbench_geometries(workspace_size)){
    oldg_strings.emplace_back(x.get_string());
  }
  

  std::vector<Geometry> newg;
  for (auto & x : allg){
    if (std::find(oldg_strings.begin(), oldg_strings.end(), x.get_string()) == oldg_strings.end()){
      newg.push_back(x);
    }
  }
  
  return newg;
}


std::vector<Geometry> get_small_deepbench_geometries(size_t small_threshold,
                                                     size_t workspace_size)
{
  auto                  all_geoms = get_deepbench_geometries(workspace_size);
  std::vector<Geometry> small_geoms;
  size_t              count_small = 0;
  for (auto& gg : all_geoms)
  {
    if (gg.m * gg.n * gg.k < small_threshold)
    {
      ++count_small;
      small_geoms.emplace_back(gg);
    }
  }
  return small_geoms;
}

std::vector<Geometry> get_large_deepbench_geometries(size_t large_threshold,
                                                     size_t workspace_size)
{
  auto                  all_geoms = get_deepbench_geometries(workspace_size);
  std::vector<Geometry> large_geoms;
  for (auto& gg : all_geoms)
  {
    if (gg.m * gg.n * gg.k >= large_threshold)
    {
      large_geoms.emplace_back(gg);
    }
  }
  return large_geoms;
}

std::vector<Geometry> get_problem_geometries()
{
  auto                  all_geoms = get_deepbench_geometries(1);
  std::vector<Geometry> large_geoms;
  for (auto& gg : all_geoms)
  {
    if (gg.m == 4096 && gg.n == 7133 && gg.k == 4096)
    {
      large_geoms.emplace_back(gg);
    }
  }
  return large_geoms;
}

std::vector<Geometry> get_backconvwrw_geometries(size_t workspace_size)
{

  // m ,  n ,  k , lda ,ldb ,ldc , tA , tB
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, bool, bool>>
    backcwrwr_problems = {

      //  m , n , k , lda , ldb , ldc , tA , tB
      std::make_tuple(100, 32, 26939, 26939, 26939, 100, true, false),
      std::make_tuple(1600, 32, 6308, 6308, 6308, 1600, true, false),
      std::make_tuple(9, 16, 23040, 23040, 23040, 9, true, false),
      std::make_tuple(144, 32, 5760, 5760, 5760, 144, true, false),
      std::make_tuple(288, 64, 1440, 1440, 1440, 288, true, false),
      std::make_tuple(576, 128, 360, 360, 360, 576, true, false),
      std::make_tuple(27, 64, 2916, 2916, 2916, 27, true, false),
      std::make_tuple(576, 64, 2916, 2916, 2916, 576, true, false),
      std::make_tuple(1152, 128, 729, 729, 729, 1152, true, false),
      std::make_tuple(1152, 256, 196, 196, 196, 1152, true, false),
      std::make_tuple(2304, 512, 49, 49, 49, 2304, true, false),
      std::make_tuple(27, 64, 50176, 50176, 50176, 27, true, false),
      std::make_tuple(576, 128, 12544, 12544, 12544, 576, true, false),
      std::make_tuple(1152, 256, 3136, 3136, 3136, 1152, true, false),
      std::make_tuple(2304, 512, 784, 784, 784, 2304, true, false),
      std::make_tuple(4608, 512, 196, 196, 196, 4608, true, false),
      std::make_tuple(4608, 512, 49, 49, 49, 4608, true, false),
      std::make_tuple(147, 64, 12544, 12544, 12544, 147, true, false),
      std::make_tuple(4800, 32, 784, 784, 784, 4800, true, false),
      std::make_tuple(192, 64, 784, 784, 784, 192, true, false),
      std::make_tuple(12800, 48, 196, 196, 196, 12800, true, false),
      std::make_tuple(512, 192, 196, 196, 196, 512, true, false),
      std::make_tuple(832, 256, 49, 49, 49, 832, true, false),
      std::make_tuple(20800, 128, 49, 49, 49, 20800, true, false),

    };

  return get_from_m_n_k_ldaABC_tA_tB(backcwrwr_problems, workspace_size);
}

std::vector<Geometry> get_small_growing_geometries(size_t workspace_size)
{

  std::vector<std::tuple<size_t, size_t, size_t, bool, bool>> scalingproblems = {
    std::make_tuple(250, 250, 50, false, false),
    std::make_tuple(250, 250, 100, false, false),
    std::make_tuple(250, 250, 200, false, false),
    std::make_tuple(250, 250, 400, false, false),
    std::make_tuple(250, 250, 800, false, false),
    std::make_tuple(250, 250, 1600, false, false),
    std::make_tuple(250, 250, 3200, false, false),
    std::make_tuple(250, 250, 6400, false, false),
    std::make_tuple(250, 250, 12800, false, false),
    std::make_tuple(250, 250, 25600, false, false),
    std::make_tuple(250, 250, 51200, false, false),
    std::make_tuple(250, 250, 102400, false, false),
  };

  return get_from_m_n_k_tA_tB(scalingproblems, workspace_size);
}

std::vector<Geometry> get_square_geometries(size_t workspace_size)
{

  std::vector<std::tuple<size_t, size_t, size_t, bool, bool>> squareproblems;

  for (size_t dim = 100; dim < 6400; dim += 100)
  {
    squareproblems.push_back(std::make_tuple(dim, dim, dim, false, false));
    squareproblems.push_back(std::make_tuple(dim, dim, dim, false, true));
    squareproblems.push_back(std::make_tuple(dim, dim, dim, true, false));
  };

  return get_from_m_n_k_tA_tB(squareproblems, workspace_size);
}
}
