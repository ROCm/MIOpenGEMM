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

/* Note that currently (13/11/2016) most testing is done through dev/python scripts */



tinygemm::hyperparams::HyperParams get_hp(){

  std::map<std::string, unsigned> hp_map;
  hp_map["micro_tile_width"] = 1;
  hp_map["micro_tile_height"] = 1;
  hp_map["macro_tile_width"] = 16;
  hp_map["macro_tile_height"] = 16; 
  hp_map["unroll"] = 16;
  
  hp_map["pad"] = 1;    
  hp_map["group_allocation"] = 1;
  hp_map["work_item_load_a_pll_to_unroll"] = 0;
  hp_map["work_item_load_b_pll_to_unroll"] = 1;
  hp_map["unroll_pragma"] = 1;
  
  hp_map["load_to_lds_interwoven"] = 0;
  hp_map["c_micro_tiles_interwoven"] = 1;
  hp_map["n_work_items_per_c_elm"] = 2;
  hp_map["unroll_for_offset"] = 0;
  hp_map["n_target_active_workgroups"] = 64;
  
  return hp_map;
}

tinygemm::TinyGemmGeometry get_geometry(){
  
  bool isColMajor = false;
  bool tA = false;
  bool tB = true;
  bool tC = false;
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
      
template <typename TFloat>     
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const tinygemm::TinyGemmGeometry & gg){
  
  
  size_t n_a = gg.lda * (gg.tA == gg.isColMajor ? gg.m : gg.k) + gg.a_offset;
  size_t n_b = gg.ldb * (gg.tB == gg.isColMajor ? gg.k : gg.n) + gg.b_offset;
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + gg.c_offset; 
  
  /* fill matrices with random floats. 
   * Note : if they're integers, the kernel runs faster! */
  v_a.resize(n_a); 
  for (size_t i = 0; i < n_a; ++i){
    v_a[i] = TFloat(rand() % 1000) / 1000. - 0.5;
  }
  
  v_b.resize(n_b);
  for (size_t i = 0; i < n_b; ++i){
    v_b[i] = TFloat(rand() % 1000) / 1000. - 0.5;
  }


  v_c.resize(n_c);
  for (size_t i = 0; i < n_c; ++i){
    v_c[i] = TFloat(rand() % 1000) / 500 - 1.;
  }
}

int main(){
  
  typedef float tfloat;
  srand(time(NULL));
  tinygemm::hyperparams::HyperParams hp = get_hp();
  tinygemm::TinyGemmGeometry gg = get_geometry();

  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;


  set_abc<tfloat>(v_a, v_b, v_c, gg);
  
  double alpha = 1.3; //1.1;
  double beta = 1.7;//1.1;
  
  const tfloat * c_true_bla = nullptr; 
  
  tinygemm::dev::accuracy_test(hp, gg, alpha, beta, v_a.data(), v_b.data(), v_c.data(), c_true_bla, true, "");

  return 0;
}

