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
    //unsigned wos = 2;
    //hyperstring = "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS" + std::to_string(wos) + "__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW0_WOS" + std::to_string(wos) + "__U8_GA1_PU0_ICE1_NAW64_UFO0";


    //hyperstring = "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE5_NAW64_UFO0";


  hyperstring = "A_MAC96_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS1__U16_GA2_PU1_ICE5_NAW64_UFO0";
  
  
    //hyperstring = "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS2__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW0_WOS2__U16_GA1_PU0_ICE1_NAW64_UFO0";
    
     //hyperstring = "A_MAC1_MIC1_PAD1_PLU0_LIW0_MIW0_WOS0__B_MAC1_MIC1_PAD1_PLU0_LIW0_MIW0_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0";
        
  }
  return hyperstring;
}

template <typename TFloat>
tinygemm::TinyGemmGeometry get_geometry(){

  //std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 35, 35, 8457, 2048, 0, 'f'}, 

  // std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 128, 3072, 0, 'f'},  

  
  bool isColMajor = true;
  bool tA = true;
  bool tB = false;
  bool tC = false;
  unsigned m = 1024;//4000;//*30;//490;//96;//640;      
  unsigned n = 128;//*30;//96;//      
  unsigned k = 3072;//                
  unsigned lda = ( tA == isColMajor ? k : m ) + 0;//10;//13;
  unsigned ldb = ( tB == isColMajor ? n : k ) + 0;//20;//27;
  unsigned ldc = ( tC == isColMajor ? n : m ) + 0;//30;//11;//11;
  unsigned workspace_size =  2e8;//150386109 ;
  char floattype = sizeof(TFloat) == sizeof(double) ? 'd' : 'f';

  return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };
    
}

tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 33;//3e6;//5;
  unsigned b_offset = 55;//2e6;//7;
  unsigned c_offset = 77;//1e6;//11;
  unsigned workspace_offset = 99;//1e6;//17;
  
  unsigned tail_off_a = 1e6 + 123;
  unsigned tail_off_b = 1e6 + 97;
  unsigned tail_off_c = 1e6 + 67;
  
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
  bool test_benchgemm = true;//true;
  bool test_find = false;
  bool test_accuracy = false;
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
    tinygemm::dev::benchgemm({hp}, 32, gg, toff, v_a.data(), v_b.data(), v_c.data(), true, "");
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

