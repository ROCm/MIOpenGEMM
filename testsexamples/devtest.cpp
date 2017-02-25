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
    //hyperstring = "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_CTY0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW0_CTY0__U8_GA1_PU0_ICE1_NAW64_UFO0";
    hyperstring = "A_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_CTY0__B_MAC96_MIC6_PAD1_PLU1_LIW0_MIW0_CTY1__U16_GA1_PU0_ICE2_NAW64_UFO0";
  }
  return hyperstring;
}

template <typename TFloat>
tinygemm::TinyGemmGeometry get_geometry(){

  
  bool isColMajor = true;
  bool tA = false;
  bool tB = true;
  bool tC = false;
  unsigned m = 961;      
  unsigned n = 961;      
  unsigned k = 161;                
  unsigned lda = ( tA == isColMajor ? k : m ) + 3;//13;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 9;//27;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;//11;//11;
  unsigned workspace_size =  1;//150386109 ;
  char floattype = sizeof(TFloat) == sizeof(double) ? 'd' : 'f';

  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 51;
  unsigned b_offset = 71;
  unsigned c_offset = 91;
  unsigned workspace_offset = 13;
  
  unsigned tail_off_a = 123;
  unsigned tail_off_b = 97;
  unsigned tail_off_c = 67;
  
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};

}


template <typename TFloat>
void print_kernel(){

  std::string kernel_string;
  auto hp = get_hp();
  auto gg = get_geometry<TFloat>();
  
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
  
  if (test_accuracy){
    tinygemm::dev::accuracy_test(hp, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, true, "");
  }

  if (test_benchgemm){
    tinygemm::dev::benchgemm({hp}, 8, gg, toff, v_a.data(), v_b.data(), v_c.data(), true, "");
  }
  
  if (test_find){
    float allotted_time = 5.;
    tinygemm::dev::find(allotted_time, v_a.data(), v_b.data(), v_c.data(), false, gg, toff, true, "");
  }
  
  if (test_default){
    auto bla = tinygemm::get_default(false, get_geometry<tfloat>(), true, "");
    for (auto & x : bla.v_tgks){
      std::cout << x.kernstr << std::endl;
    }
  }
  
  return 0;
}

