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
    //hyperstring = "A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC5";        
    hyperstring = "A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC5";
  }
  return hyperstring;
}

template <typename TFloat>
tinygemm::TinyGemmGeometry get_geometry(){

  bool goodsolly = false;  
  bool isColMajor = true;
  bool tA = false;
  bool tB = true;
  bool tC = false;
  unsigned m = 128*(32) - 6; 
  unsigned n = 96*(55) - 4; 
  unsigned k = 16*229;           

  if (goodsolly == false){
    isColMajor = true;
    tA = false;
    tB = true;
    tC = false;
    m = 2560;
    n = 33;//7133;
    k = 2560;
  }    
  
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
  
  std::cout << "in main of devtest " << std::endl;

  tinygemm::FindStartType fst(tinygemm::FindStartType::Random);

  bool test_print = false;
  bool test_benchgemm = false;
  bool test_find = true;
  bool test_accuracy = false;
  bool test_default = false;
  
  typedef float tfloat;
  srand(time(NULL));
  
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
      tinygemm::dev::benchgemm({hp}, 7, gg, toff, v_a.data(), v_b.data(), v_c.data(), true, "");
    }
  }
  
  if (test_find){
    std::string constraint_string("");
    float allotted_time = 0.1;
    tinygemm::dev::find(allotted_time, v_a.data(), v_b.data(), v_c.data(), constraint_string, fst, gg, toff, true, "/home/james/output.txt");
  }
  
  if (test_default){
    throw tinygemm::tinygemm_error("cannot test default currently, bla");
  }
  
  return 0;
}


void bingbing(){
  auto blabla =  
  
  {

  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 128, 3072, 0, 'f'},  
  "A_MAC96_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U32_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 32, 1024, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 16, 7680, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U32_GA1_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2560, 5124, 5124, 9124, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2560, 7133, 2560, 2560, 7133, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 'f'},  "A_MAC8_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2560, 2560, 8457, 35, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW1_MIW1_WOS0__U8_GA2_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7680, 5481, 7680, 7680, 5481, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2048, 2048, 8457, 35, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW1_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW1_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 1760, 1760, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2048, 35, 35, 8457, 2048, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 4096, 5124, 5124, 9124, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU1_LIW1_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU1_ICE5_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2560, 35, 35, 8457, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 32, 7680, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 4096, 4096, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1760, 7133, 1760, 1760, 7133, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4096, 7133, 4096, 4096, 7133, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 5124, 5124, 9124, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA3_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 128, 1024, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U48_GA2_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 64, 2560, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 128, 1024, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 64, 3072, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 35, 35, 8457, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 32, 2560, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU1_LIW1_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU1_ICE9_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 4096, 35, 35, 8457, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 32, 1024, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 1760, 35, 35, 8457, 1760, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 4096, 4096, 8457, 35, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U64_GA3_PU1_ICE8_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 35, 35, 8457, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 64, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA3_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 16, 1024, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE9_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 5124, 5124, 9124, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 35, 35, 8457, 2048, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2560, 2560, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 16, 3072, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U32_GA2_PU0_ICE9_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2048, 2048, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 16, 1024, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU1_LIW1_MIW1_WOS0__U32_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3072, 7435, 3072, 3072, 7435, 1024, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 64, 7680, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE9_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 1760, 1760, 8457, 35, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC80_MIC5_PAD1_PLU1_LIW1_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 128, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 5124, 5124, 9124, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA3_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 64, 1024, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U48_GA2_PU0_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 32, 3072, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 16, 2560, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE8_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 5124, 5124, 9124, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE8_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 16, 2560, 0, 'f'},  "A_MAC40_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE2_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__U32_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 1760, 5124, 5124, 9124, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 32, 2560, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2048, 7133, 2048, 2048, 7133, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 35, 35, 8457, 1760, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 64, 1024, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 128, 7680, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2048, 5124, 5124, 9124, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 128, 3072, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE8_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 32, 1024, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 16, 7680, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U64_GA2_PU1_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2567, 5137, 5124, 9124, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2565, 7140, 2573, 2560, 7133, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2573, 2560, 8457, 35, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA3_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7685, 5488, 7693, 7680, 5481, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2061, 2048, 8457, 35, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 1773, 1760, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2055, 48, 35, 8457, 2048, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 4103, 5137, 5124, 9124, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2567, 48, 35, 8457, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 32, 7680, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 4109, 4096, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1765, 7140, 1773, 1760, 7133, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4101, 7140, 4109, 4096, 7133, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 5137, 5124, 9124, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA3_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 128, 1024, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW1_MIW1_WOS0__U16_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 64, 2560, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 128, 1024, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 64, 3072, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 48, 35, 8457, 2560, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 32, 2560, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 'f'},  "A_MAC8_MIC1_PAD1_PLU0_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U32_GA1_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA3_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 32, 1024, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 1767, 48, 35, 8457, 1760, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW1_MIW0_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW1_MIW0_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 4103, 48, 35, 8457, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 128, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 'f'},  "A_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 48, 35, 8457, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 64, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE3_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW1_MIW1_WOS0__U16_GA3_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 16, 1024, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U32_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 5137, 5124, 9124, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA3_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 48, 35, 8457, 2048, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2573, 2560, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 16, 3072, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2061, 2048, 9124, 5124, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA3_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 16, 1024, 0, 'f'},  "A_MAC16_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3077, 7442, 3085, 3072, 7435, 1024, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 64, 7680, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 48, 35, 8457, 1760, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC40_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 4109, 4096, 8457, 35, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 5137, 5124, 9124, 2048, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 64, 1024, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 32, 3072, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 16, 2560, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U48_GA2_PU1_ICE8_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA3_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U48_GA1_PU1_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 5137, 5124, 9124, 4096, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA3_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 'f'},  "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 'f'},  "A_MAC24_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 16, 2560, 0, 'f'},  "A_MAC40_MIC5_PAD1_PLU1_LIW1_MIW1_WOS0__B_MAC16_MIC2_PAD1_PLU0_LIW1_MIW1_WOS0__U8_GA1_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 1767, 5137, 5124, 9124, 1760, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 32, 2560, 0, 'f'},  "A_MAC48_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2053, 7140, 2061, 2048, 7133, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO1"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 1773, 1760, 8457, 35, 0, 'f'},  "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 64, 1024, 0, 'f'},  "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 128, 7680, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 'f'},  "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE4_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2055, 5137, 5124, 9124, 2048, 0, 'f'},  "A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0__U8_GA1_PU0_ICE1_NAW64_UFO0"), 
  
  //some back conv problems : 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1000, 1000, 16, 16, 16, 1000, 0, 'f'},    "A_MAC8_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC8_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__U40_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5760, 5760, 144, 144, 32, 5760, 0, 'f'},    "A_MAC8_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC8_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 23040, 23040, 9, 9, 16, 23040, 0, 'f'},    "A_MAC8_MIC1_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC8_MIC1_PAD1_PLU1_LIW0_MIW0_WOS0__U48_GA1_PU1_ICE37_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 12544, 12544, 147, 147, 64, 12544, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU1_ICE24_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 26939, 26939, 100, 100, 32, 26939, 0, 'f'},    "A_MAC16_MIC1_PAD1_PLU0_LIW0_MIW0_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U32_GA1_PU1_ICE18_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2916, 2916, 27, 27, 64, 2916, 0, 'f'},    "A_MAC8_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC8_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U32_GA2_PU0_ICE12_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 50176, 50176, 27, 27, 64, 50176, 0, 'f'},    "A_MAC16_MIC1_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U48_GA2_PU1_ICE32_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 832, 832, 256, 49, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA2_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 192, 192, 64, 784, 0, 'f'},    "A_MAC16_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2916, 2916, 576, 576, 64, 2916, 0, 'f'},    "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE7_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 6308, 6308, 1600, 1600, 32, 6308, 0, 'f'},    "A_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U40_GA1_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1440, 1440, 288, 288, 64, 1440, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU0_ICE6_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 4608, 4608, 512, 49, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 2304, 2304, 512, 49, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 512, 512, 192, 196, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC16_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 49, 49, 20800, 20800, 128, 49, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW0_WOS0__U8_GA3_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 360, 360, 576, 576, 128, 360, 0, 'f'},    "A_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__B_MAC16_MIC2_PAD1_PLU1_LIW0_MIW0_WOS0__U24_GA2_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 4608, 4608, 512, 196, 0, 'f'},    "A_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU1_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 12800, 12800, 48, 196, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3136, 3136, 1152, 1152, 256, 3136, 0, 'f'},    "A_MAC80_MIC5_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA1_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 196, 196, 1152, 1152, 256, 196, 0, 'f'},    "A_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE1_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 12544, 12544, 576, 576, 128, 12544, 0, 'f'},    "A_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA3_PU0_ICE20_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 4800, 4800, 32, 784, 0, 'f'},    "A_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU0_LIW0_MIW1_WOS0__U32_GA1_PU1_ICE5_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 729, 729, 1152, 1152, 128, 729, 0, 'f'},    "A_MAC48_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC32_MIC2_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA2_PU0_ICE2_NAW64_UFO0"), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 784, 784, 2304, 2304, 512, 784, 0, 'f'},    "A_MAC96_MIC6_PAD1_PLU1_LIW0_MIW1_WOS0__B_MAC64_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE1_NAW64_UFO0"), 
  
};

}
