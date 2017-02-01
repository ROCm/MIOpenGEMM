#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/kernelstringgenerator.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>

#include "setabc.hpp"
/* Note that currently (13/11/2016) most testing is done through dev/python scripts */
/* Update (30/01/2017) this is the prefered place for doing testing */



tinygemm::hyperparams::HyperParams get_hp(std::string hyperstring = ""){

  if (hyperstring.compare("") == 0){
    hyperstring = "Y16_X8_y2_x1_U16_P1_GA3_APLU0_BPLU1_PU0_LIW1_MIW1_ICE4_NAW64_UFO1";
  }
  
  return hyperstring;
      
}

tinygemm::TinyGemmGeometry get_geometry(){
  
  bool isColMajor = true;
  bool tA = false;
  bool tB = true;
  bool tC = true;
  unsigned m = 418;    
  unsigned n = 100;
  unsigned k = 123;
  unsigned lda = ( tA == isColMajor ? k : m ) + 1;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 2;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 3;
  unsigned a_offset = 5;
  unsigned b_offset = 7;
  unsigned c_offset = 11;

  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset };
    
}
      


void print_kernel(){

  std::string kernel_string;
  auto hp = get_hp();
  auto gg = get_geometry();
  
  auto bundle = tinygemm::kerngen::get_kernel_string_bundle(
  hp,  
//  kernel_string,
  "bolziberb",
  32,
  gg
  
  );
  
  std::cout << bundle.kernel_string;
  
  std::ofstream floper ("/home/idiap/akernel.cl", std::ios::out); 
  floper << bundle.kernel_string;
  
  floper.close();
}
  


int main(){
  

  bool test_print = true;
  bool test_benchgemm = false;
  bool test_find = false;
  bool test_accuracy = true;
  
  typedef float tfloat;
  srand(time(NULL));
  tinygemm::hyperparams::HyperParams hp = get_hp();
  
  tinygemm::TinyGemmGeometry gg = get_geometry();

  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;


  setabc::set_abc<tfloat>(v_a, v_b, v_c, gg);
  
  double alpha = 1.3; 
  double beta = 1.7;
  
  const tfloat * c_true_bla = nullptr; 
  
  if (test_print){
    print_kernel();
  }
  
  if (test_accuracy){
    tinygemm::dev::accuracy_test(hp, gg, alpha, beta, v_a.data(), v_b.data(), v_c.data(), c_true_bla, true, "");
  }

  if (test_benchgemm){
    tinygemm::dev::benchgemm({hp}, 5, gg, alpha, beta, v_a.data(), v_b.data(), v_c.data(), true, "");
  }
  
  if (test_find){
    float allotted_time = 5.;
    tinygemm::dev::find(allotted_time, v_a.data(), v_b.data(), v_c.data(), false, gg, alpha, beta, true, "");
  }
  
  return 0;
}

