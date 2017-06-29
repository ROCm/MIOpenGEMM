#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cblas.h>

#include <miopengemm/kernelcache.hpp>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/openclutil.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/basicfind.hpp>

MIOpenGEMM::Offsets get_offsets(){
  unsigned a_offset = 0;
  unsigned b_offset = 0;
  unsigned c_offset = 0;
  unsigned workspace_offset = 0;
  unsigned tail_off_a = 0;
  unsigned tail_off_b = 0;
  unsigned tail_off_c = 0;
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};
}



template <typename TFloat>
int go(){

  std::string fout("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(false, fout != "" , fout);
  
  MIOpenGEMM::openclutil::CommandQueueInContext tgcq(mowri, "in benchthecache.cpp");
  MIOpenGEMM::openclutil::OpenCLDeviceInfo devinfo(tgcq.command_queue);
 
 
  unsigned counter = 0;
  
  for (auto & x : MIOpenGEMM::kernel_cache){
    auto identifier = x.first;
    if (identifier == devinfo.identifier){
      std::cout << "\nCACHE DEIVCE ID: " << identifier << std::endl;
      for (auto & y : x.second){
        auto constraints_string = y.first;
        std::cout << "\nCONSTRAINTS STRING: " << constraints_string<< std::endl;
        for (auto & z : y.second){
          auto geometry_string = z.first;
          std::cout << "\nGEOMETRY STRING: " << geometry_string<< std::endl;
          for (auto & a : z.second){
            auto comment_string = a.first;
            MIOpenGEMM::Geometry gg(geometry_string);          
            MIOpenGEMM::Offsets toff = get_offsets();          
            
            if (gg.derived.float_size_bytes == sizeof(TFloat)){
              
              
              auto soln2 = MIOpenGEMM::kernel_cache.at(identifier).at(constraints_string).at(geometry_string).at(comment_string);
              auto soln1 = MIOpenGEMM::get_default(tgcq.command_queue, constraints_string, gg, comment_string, mowri);

              std::vector<TFloat> v_a;
              std::vector<TFloat> v_b;
              std::vector<TFloat> v_c;

              MIOpenGEMM::setabcw::set_abc<TFloat>(v_a, v_b, v_c, gg, toff);
              std::vector<TFloat> v_c_final_true(v_c);

              if (gg.tX[MIOpenGEMM::nsHP::matC]){
                throw MIOpenGEMM::miog_error("gg tC not supp");
              }
              
              if (soln1.hyper_param_string != soln2.hyperstring){
                throw MIOpenGEMM::miog_error("path to default soln broken?");
              }
              
              ++counter;


  MIOpenGEMM::FindParams find_params(0.01, 1, 4, MIOpenGEMM::Max);


      auto soln = MIOpenGEMM::basicfind(gg,
                            toff,
                            find_params,
                            false,
                            "",
                            soln1.hyper_param_string,
                            0,
                            false);

      std::cout << "soln median gflops :  " << soln.statistics.median_benchmark_gflops
                  << "  \t soln median time : " << soln.statistics.median_benchmark_time << std::endl;

            }
          }
        }
      }
    }
  }
  return 0;
}

int main(){
  go<float>();
  

}
