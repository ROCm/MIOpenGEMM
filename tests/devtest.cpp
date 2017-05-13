#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/bundle.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/outputwriter.hpp>

#include <tinygemm/setabcw.hpp>
/* Note that currently (13/11/2016) most testing is done through dev/python scripts */
/* Update (30/01/2017) this is the preferred place for doing testing */


std::string get_hyperstring(std::string hyperstring = ""){
  if (hyperstring.compare("") == 0){
    //hyperstring = "A_MIC5_PAD2_PLU0_LIW1_MIW0_WOS1__B_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__C_UNR32_GAL3_PUN1_ICE1_NAW64_UFO0_MAC2";
    hyperstring = "A_MIC4_PAD2_PLU1_LIW1_MIW1_WOS0__B_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE3_NAW64_UFO0_MAC256_SKW9";
  }
  return hyperstring;
}

template <typename TFloat>
tinygemm::TinyGemmGeometry get_geometry(){


  //m1760n64k1760tA1tB0

  bool isColMajor = true;
  bool tA = false;
  bool tB = false;
  bool tC = false;
  unsigned m = 200; 
  unsigned n = 500; 
  unsigned k = 10000;           

  
  unsigned lda = ( tA == isColMajor ? k : m ) + 0;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;
  unsigned workspace_size =  1e1;
  char floattype = sizeof(TFloat) == sizeof(double) ? 'd' : 'f';
  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 0;//33;//3e6;//5;
  unsigned b_offset = 0;//55;//2e6;//7;
  unsigned c_offset = 0;//77;//1e6;//11;
  unsigned workspace_offset = 0;//99;//1e6;//17;
  unsigned tail_off_a = 0;//1e6 + 123;
  unsigned tail_off_b = 0;//1e6 + 97;
  unsigned tail_off_c = 0;//1e6 + 67;
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};

}


template <typename TFloat>
void print_kernel(){
  std::string kernel_string;  
  std::string hyperstring = get_hyperstring();
  auto gg = get_geometry<TFloat>();

  tinygemm::openclutil::OpenCLDeviceInfo devinfo;
  tinygemm::hyperparams::Graph graph(gg, devinfo, hyperstring, true); 
  tinygemm::hyperparams::HyperParams hp(graph);
  bool mowri_verbose = true;
  bool verbose_get_bundle = true;
  std::string mowri_out("");
  tinygemm::outputwriting::OutputWriter mowri(mowri_verbose, mowri_out != "" , mowri_out);
  auto bundle = tinygemm::kerngen::get_bundle(hp, gg, mowri, verbose_get_bundle);
  
  
  for (auto & x :  bundle.v_tgks){
    auto fname = "/home/james/akernel_" +  x.type.full +  ".cl";
    std::cout << "writing " << fname << " ... " << std::flush;
    std::ofstream floper (fname, std::ios::out); 
    floper << x.kernstr;
    floper.close();
    std::cout << "done." << std::endl;
  }
}
  


int main(){

  std::string fout("");
  tinygemm::outputwriting::OutputWriter mowri(true, fout != "" , fout);


  bool test_print = false;
  bool test_benchgemm = true;  
  bool test_find = false;
  bool test_accuracy = true;
  bool test_default = false;

  std::string constraints_string("");
  
  float allotted_find_time = 1000;
  unsigned allotted_find_descents = 10;
  unsigned n_runs_per_kernel = 3;
  tinygemm::SummaryStat sumstat(tinygemm::Max);
  
  unsigned n_runs_benchgemm = 2;
  
  typedef float tfloat;
  
  
  tinygemm::TinyGemmGeometry gg = get_geometry<tfloat>();
  tinygemm::TinyGemmOffsets toff = get_offsets();
  mowri << "generating cpu data ... " << tinygemm::Flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  mowri << "done." << tinygemm::Endl;
  const tfloat * c_true_bla = nullptr; 
  if (test_print){
    print_kernel<tfloat>();
  }
  
  if (test_accuracy || test_benchgemm){
    std::string hyperstring = get_hyperstring();
    if (test_accuracy){
      tinygemm::dev::accuracy_test(hyperstring, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
    }

    if (test_benchgemm){
      tinygemm::dev::benchgemm({hyperstring}, n_runs_benchgemm, gg, toff, v_a.data(), v_b.data(), v_c.data(), mowri);
    }
  }
  
  
  tinygemm::FindParams find_params(allotted_find_time, allotted_find_descents, n_runs_per_kernel, sumstat);
  
  if (test_find){
    tinygemm::dev::find(find_params, v_a.data(), v_b.data(), v_c.data(), constraints_string, gg, toff, mowri);
  }
  
  if (test_default){
    std::string k_comment("");  
    tinygemm::openclutil::TinyGemmCommandQueueInContext tgcq(mowri, "TinyGemmCommandQueueInContext in get_default in devtingemm.cpp");
    auto soln = tinygemm::get_default(tgcq.command_queue, constraints_string, gg, k_comment, mowri);
  
    //throw tinygemm::tinygemm_error("cannot test default currently, bla");
  }
  
  return 0;
}


