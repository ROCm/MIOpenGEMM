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
/* Update (30/01/2017) this is the prefered place for doing testing */



tinygemm::hyperparams::HyperParams get_hp(std::string hyperstring = ""){

  if (hyperstring.compare("") == 0){
    hyperstring = "Y64_X64_y4_x4_U8_P1_GA3_APLU1_BPLU1_PU0_LIW1_MIW1_ICE2_NAW64_UFO0_ACW0_BCW0";
    //hyperstring = "Y128_X96_y8_x6_U8_P1_GA3_APLU0_BPLU1_PU1_LIW1_MIW1_ICE2_NAW64_UFO1_NAW64_UFO0";
  }
  return hyperstring;

}

tinygemm::TinyGemmGeometry get_geometry(){
  
  bool isColMajor = true;
  bool tA = false;
  bool tB = true;
  bool tC = false;
  unsigned m = 1096;
  unsigned n = 1096;
  unsigned k = 1096;
  unsigned lda = ( tA == isColMajor ? k : m ) + 13;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 27;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 13;//11;
  unsigned workspace_size = 15;
  char floattype = 'f';

  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype};
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 51;
  unsigned b_offset = 71;
  unsigned c_offset = 91;
  unsigned workspace_offset = 13;
  return { a_offset, b_offset, c_offset, workspace_offset };

}


void print_kernel(){

  std::string kernel_string;
  auto hp = get_hp();
  auto gg = get_geometry();
  
  auto bundle = tinygemm::kerngen::get_bundle(hp, gg);
  
  //std::cout << bundle.v_tgks[0].kernstr; //.back()
  

    //std::ofstream floper ("/home/idiap/tinygemm/examplekernels/example1.cl", std::ios::out); 
  
  for (auto & x :  bundle.v_tgks){
    std::ofstream floper ("/home/idiap/akernel_" +  x.type+ ".cl", std::ios::out); 
    floper << x.kernstr;
    floper.close();
  }

}
  


int main(){
  

  bool test_print = true;
  bool test_benchgemm = false;//true;
  bool test_find = false;
  bool test_accuracy = false;
  bool test_default = false;
  
  typedef float tfloat;
  srand(time(NULL));
  tinygemm::hyperparams::HyperParams hp = get_hp();
  
  tinygemm::TinyGemmGeometry gg = get_geometry();
  tinygemm::TinyGemmOffsets toff = get_offsets();



  std::cout << "generating cpu data ... " << std::flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  std::cout << "done." << std::endl;
    
  const tfloat * c_true_bla = nullptr; 
  
  if (test_print){
    print_kernel();
  }
  
  if (test_accuracy){
    tinygemm::dev::accuracy_test(hp, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, true, "");
  }

  if (test_benchgemm){
    tinygemm::dev::benchgemm({hp}, 5, gg, toff, v_a.data(), v_b.data(), v_c.data(), true, "");
  }
  
  if (test_find){
    float allotted_time = 5.;
    tinygemm::dev::find(allotted_time, v_a.data(), v_b.data(), v_c.data(), false, gg, toff, true, "");
  }
  
  if (test_default){
    auto bla = tinygemm::get_default(false, get_geometry(), true, "");
    for (auto & x : bla.v_tgks){
      std::cout << x.kernstr << std::endl;
    }
  }
  
  return 0;
}

