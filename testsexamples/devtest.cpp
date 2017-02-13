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

hyperstring = "Y8_X8_y1_x1_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ICE3_NAW64_UFO0_ACW0_BCW0";
    //hyperstring = "Y64_X64_y4_x4_U8_P1_GA3_APLU1_BPLU1_PU0_LIW1_MIW1_ICE2_NAW64_UFO0_ACW1_BCW1";





    //hyperstring = "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU1_LIW1_MIW1_ICE1_NAW64_UFO0_ACW1_BCW1";
  }
  return hyperstring;

}

tinygemm::TinyGemmGeometry get_geometry(){


//tC0_tA0_tB0_colMaj1_m16_n64_k800_lda16_ldb800_ldc16
  
  bool isColMajor = true;
  bool tA = false;
  bool tB = false;
  bool tC = false;
  unsigned m = 16;//4096;
  unsigned n = 64;//5025;
  unsigned k = 800;//4096;
  unsigned lda = ( tA == isColMajor ? k : m ) + 0;//13;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;//27;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;//13;//11;
  unsigned workspace_size =  38386109 ;
  char floattype = 'f';

  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 51;
  unsigned b_offset = 71;
  unsigned c_offset = 91;
  unsigned workspace_offset = 13;
  
  unsigned tail_off_a = 12345;
  unsigned tail_off_b = 12345;
  unsigned tail_off_c = 12345;
  
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};

}


void print_kernel(){

  std::string kernel_string;
  auto hp = get_hp();
  auto gg = get_geometry();
  
  auto bundle = tinygemm::kerngen::get_bundle(hp, gg);
  
  //std::cout << bundle.v_tgks[0].kernstr; //.back()
  

    //std::ofstream floper ("/home/idiap/tinygemm/examplekernels/example1.cl", std::ios::out); 
  
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
  

  bool test_print = true;
  bool test_benchgemm = false;//true;
  bool test_find = false;
  bool test_accuracy = true;
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

