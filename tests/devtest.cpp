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

#include "setabcw.hpp"
/* Note that currently (13/11/2016) most testing is done through dev/python scripts */
/* Update (30/01/2017) this is the preferred place for doing testing */


std::string get_hyperstring(std::string hyperstring = ""){
  if (hyperstring.compare("") == 0){
    hyperstring = "A_MIC3_PAD0_PLU1_LIW1_MIW1_WOS0__B_MIC2_PAD2_PLU0_LIW0_MIW0_WOS0__C_UNR32_GAL2_PUN1_ICE3_NAW16_UFO0_MAC5";
  }
  return hyperstring;
}

template <typename TFloat>
tinygemm::TinyGemmGeometry get_geometry(){

  //bool goodsolly = false;  
  bool isColMajor = true;
  bool tA = false;
  bool tB = false;
  bool tC = false;
  unsigned m = 1760;//128*(32) - 6; 
  unsigned n = 32;//96*(55) - 4; 
  unsigned k = 1760;//16*229;           

  //if (goodsolly == false){
    //isColMajor = true;
    //tA = false;
    //tB = true;
    //tC = false;
    //m = 2560;
    //n = 33;//7133;
    //k = 2560;
  //}    
  
  unsigned lda = ( tA == isColMajor ? k : m ) + 0;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;
  unsigned workspace_size =  1;
  char floattype = sizeof(TFloat) == sizeof(double) ? 'd' : 'f';
  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 33;//3e6;//5;
  unsigned b_offset = 55;//2e6;//7;
  unsigned c_offset = 77;//1e6;//11;
  unsigned workspace_offset = 99;//1e6;//17;
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
  tinygemm::hyperparams::Graph graph(gg, hyperstring, true); 
  tinygemm::hyperparams::HyperParams hp(graph);
  auto bundle = tinygemm::kerngen::get_bundle(hp, gg);
  
  for (auto & x :  bundle.v_tgks){
    auto fname = "/home/james/dub_akernel_" +  x.type.full +  ".cl";
    std::cout << "writing " << fname << " ... " << std::flush;
    std::ofstream floper (fname, std::ios::out); 
    floper << x.kernstr;
    floper.close();
    std::cout << "done." << std::endl;
  }
}
  


int main(){

  bool test_print = false;
  bool test_benchgemm = true;  
  bool test_find = false;
  bool test_accuracy = false;
  bool test_default = false;

  std::string constraint_string("");
  float allotted_find_time = 2.1;
  unsigned n_runs_benchgemm = 1000;
  
  typedef float tfloat;
  
  
  tinygemm::TinyGemmGeometry gg = get_geometry<tfloat>();
  tinygemm::TinyGemmOffsets toff = get_offsets();
  std::cout << "generating cpu data ... " << std::flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  std::cout << "done." << std::endl;
  const tfloat * c_true_bla = nullptr; 
  if (test_print){
    print_kernel<tfloat>();
  }
  
  if (test_accuracy || test_benchgemm){
    std::string hyperstring = get_hyperstring();
    tinygemm::hyperparams::Graph graph(gg, hyperstring, true); 
    tinygemm::hyperparams::HyperParams hp(graph);
    if (test_accuracy){
      tinygemm::dev::accuracy_test(hp, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, true, "");
    }

    if (test_benchgemm){
      tinygemm::dev::benchgemm({hp}, n_runs_benchgemm, gg, toff, v_a.data(), v_b.data(), v_c.data(), true, "");
    }
  }
  
  if (test_find){
    tinygemm::FindStartType fst(tinygemm::FindStartType::Random);
    tinygemm::dev::find(allotted_find_time, v_a.data(), v_b.data(), v_c.data(), constraint_string, fst, gg, toff, true, "/home/james/output.txt");
  }
  
  if (test_default){
    throw tinygemm::tinygemm_error("cannot test default currently, bla");
  }
  
  return 0;
}


