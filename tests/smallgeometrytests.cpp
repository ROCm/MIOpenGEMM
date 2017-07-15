/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/miogemm.hpp>

template <typename TFloat>
void geometrytest(bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k)
{

  using namespace MIOpenGEMM;
  Offsets offsets = get_padding_offsets(); 
  size_t  workspace_size   = 30000000; // TODO : checks on workspace size
  Geometry gg = get_padded_geometry<TFloat> (isColMajor, tA, tB, tC, m, n, k, workspace_size); 

  //FindParams find_params = get_quick_find_params();
  
  FindParams find_params(1, 1.00001, 1, 200., SummStat::E::MEDIAN);


  std::string constraints_string = "";//"A_WOS1__B_WOS2";// "C_ICE5";//A_MIC1_PAD1_PLU0_LIW0_MIW1_WOS1__B_MIC4_PAD1_PLU1_LIW0_MIW1_WOS1__C_UNR8_GAL2_PUN1_ICE1_NAW16_UFO0_MAC64_SKW9";

  // TODO : tests on ICE. if ICE =111, floating point exception.
  // TODO : test the constraints! 
  
  //(benchgemm) hp   :
//(benchgemm) geometry  	:tC0_tA1_tB0_colMaj0_m55_n65_k124_lda64_ldb75_ldc77_ws300000_f64


  owrite::Writer mowri(Ver::E::ACCURACY, "");
  dev::Boa boa(gg, offsets, mowri);  
  Solution soln = boa.find(find_params, constraints_string);
  
  std::cout << "\nWill check " << soln.hypas.get_string() << '\n';
  boa.accuracy_test(soln.hypas);
  
  
  //owrite::Writer mowri2(Ver::E::TERMINAL, "");
  //dev::Boa boa2(gg, offsets, mowri);  
  //boa2.accuracy_test(soln.hypas);


  
}



int main()
{



  //while (true){
    //using namespace MIOpenGEMM;  
    //HyPas hypas("A_MIC1_PAD1_PLU0_LIW0_MIW1_WOS1__B_MIC3_PAD2_PLU0_LIW0_MIW1_WOS2__C_UNR16_GAL2_PUN1_ICE1_NAW16_UFO0_MAC64_SKW10");
    //Geometry gg("tC0_tA0_tB1_colMaj0_m45_n55_k43_lda52_ldb53_ldc67_ws30000000_f32");
    //Offsets offsets = get_padding_offsets(); 
    //owrite::Writer mowri(Ver::E::ACCURACY, "");
    //dev::Boa boa(gg, offsets, mowri);  
    //boa.accuracy_test(hypas);
  //}
  
    
    

  size_t m     = 45;
  size_t k     = 39;
  size_t testi = 0;
  for (bool tC : {false, true})
  {
    for (bool isColMajor : {false, true})
    {
      for (bool tA : {false, true})
      {
        for (bool tB : {false, true})
        {
          for (size_t n : {m - 10, m + 10})
          {
            testi += 1;
            k += 1;
            std::cout << "\n\ntest " << testi << "/32";
            std::cout << "\nm=" << m << " n=" << n << " k=" << k << "\ntA=" << tA << " tB=" << tB
                      << " tC=" << tC << " isColMajor=" << isColMajor << std::endl;
            
            std::cout << "<float>  ";
            geometrytest<float>(isColMajor, tA, tB, tC, m, n, k);
            
            std::cout << "\n<double> ";
            geometrytest<double>(isColMajor, tA, tB, tC, m, n, k);
          }
        }
      }
    }
  }
  
  
  return 0;
}
