/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/

#include <cblas.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <miopengemm/basicfind.hpp>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/openclutil.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/iterexperiments.hpp>
#include <miopengemm/stringutilbase.hpp>

MIOpenGEMM::Offsets get_offsets()
{
  size_t a_offset         = 0;
  size_t b_offset         = 0;
  size_t c_offset         = 0;
  size_t workspace_offset = 0;
  size_t tail_off_a       = 0;
  size_t tail_off_b       = 0;
  size_t tail_off_c       = 0;
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};
}

template <typename TFloat>
int go(bool only_deepbench)
{

  std::string                                   fout("");
  MIOpenGEMM::outputwriting::OutputWriter       mowri(false, fout != "", fout);
  MIOpenGEMM::openclutil::CommandQueueInContext tgcq(mowri, "in benchthecache.cpp");
  MIOpenGEMM::openclutil::OpenCLDeviceInfo      devinfo(tgcq.command_queue);
  size_t                                      counter = 0;


  auto db_geoms = MIOpenGEMM::get_deepbench_geometries(1);
  std::vector<std::string> db_geom_strings;
  for (auto & x : db_geoms){
    db_geom_strings.push_back(x.get_string());
  }

  
  
  size_t n_to_bench = 0;
  MIOpenGEMM::Offsets  toff = get_offsets();
  std::vector<MIOpenGEMM::Geometry> all_geometries;
  for (auto& x : MIOpenGEMM::kernel_cache)
  {
    auto identifier = x.first;
    if (identifier == devinfo.identifier)
    {
      for (auto& y : x.second)
      {
        auto constraints_string = y.first;
        for (auto& z : y.second)
        {
          auto geometry_string = z.first;
          
          if (only_deepbench && std::find(db_geom_strings.begin(), db_geom_strings.end(), geometry_string) == db_geom_strings.end()){
            //do nothing
          }
          else{
            all_geometries.emplace_back(geometry_string);
            ++n_to_bench;
          }
        }
      }
    }
  }
    
  std::vector<TFloat> v_a;
  std::vector<TFloat> v_b;
  std::vector<TFloat> v_c;  
  MIOpenGEMM::setabcw::set_multigeom_abc<TFloat>(v_a, v_b, v_c, all_geometries, toff);
  std::vector<TFloat> v_c_final_true(v_c);



  for (auto& x : MIOpenGEMM::kernel_cache)
  {
    auto identifier = x.first;
    if (identifier == devinfo.identifier)
    {
      std::cout << "\nCACHE DEIVCE ID: " << identifier << std::endl;
      for (auto& y : x.second)
      {
        auto constraints_string = y.first;
        std::cout << "CONSTRAINTS STRING: " << constraints_string << "\n" << std::endl;
        for (auto& z : y.second)
        {
          auto geometry_string = z.first;
          MIOpenGEMM::Geometry gg(geometry_string);
          
          
          if (only_deepbench && std::find(db_geom_strings.begin(), db_geom_strings.end(), gg.get_string()) == db_geom_strings.end()){
            continue;
          }
          
    
          for (auto& a : z.second)
          {
            auto comment_string = a.first;
            


            if (gg.derived.float_size_bytes == sizeof(TFloat))
            {

              auto soln2 = MIOpenGEMM::kernel_cache.at(identifier)
                             .at(constraints_string)
                             .at(geometry_string)
                             .at(comment_string);
              auto soln1 = MIOpenGEMM::get_default(
                tgcq.command_queue, constraints_string, gg, comment_string, mowri);

              if (gg.tX[MIOpenGEMM::Mat::E::C])
              {
                throw MIOpenGEMM::miog_error("gg tC not supp");
              }

              if (soln1.hyper_param_string != soln2.hyperstring)
              {
                throw MIOpenGEMM::miog_error("path to default soln broken?");
              }

              ++counter;

              MIOpenGEMM::FindParams find_params(0.01, 1, 4, MIOpenGEMM::SummStat::E::MAX);

              bool use_mowri_tracker = false;

              auto soln = MIOpenGEMM::basicfind(
                gg, toff, find_params, false, "", soln1.hyper_param_string, 0, false, use_mowri_tracker);


              std::cout << MIOpenGEMM::stringutil::get_char_padded(counter, 2) << "/" << n_to_bench << "  " << gg.get_tabbed_string()
                        << "\t  gflops: " << soln.statistics.median_benchmark_gflops
                        << "    time[ms]: " << soln.statistics.median_benchmark_time << std::endl;
            }
          }
        }
      }
    }
  }
  return 0;
}

int main() {
  bool only_deepbench = true;
  go<float>(only_deepbench); 
}
