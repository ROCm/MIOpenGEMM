/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
  

int main(){

  using namespace MIOpenGEMM;
  
  bool test_accuracy_of_soln = false;
  bool bench_the_soln = false;
  
  
  Geometry gg("tC0_tA0_tB0_colMaj0_m28_n2048_k2048_lda2048_ldb2048_ldc2048_ws400000000_f32");
  CLHint devhint;
  Offsets offsets = get_padding_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::Boa boa(gg, offsets, mowri, devhint);  

  // FindParams find_params = get_quick_find_params();   
  FindParams find_params(100, 1., 3, 1., SummStat::E::MAX);
  Constraints constraints("A_WOS2__B_WOS2");
  Solution soln = boa.find(find_params, constraints);

  if (test_accuracy_of_soln){
    mowri << "\n\n\nAccuracy\n";
    boa.accuracy_test(soln.hypas);
  }
  
  if (bench_the_soln){
    mowri << "\n\n\nBenchmark\n";
    boa.benchgemm({soln.hypas}, {11, 100.});
  }
  
  return 0;
}
