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
  size_t  workspace_size   = 300000;
  Geometry gg = get_padded_geometry<TFloat>(isColMajor, tA, tB, tC, m, n, k, workspace_size); 

  //FindParams find_params = get_quick_find_params();
  
  FindParams find_params(1, 10., 1, 10., SummStat::E::MEDIAN);


  std::string constraints_string = "A_WOS1__B_WOS1__C_ICE3";
  
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  
  dev::Boa boa(gg, offsets, mowri);
  Solution soln = boa.find(find_params, constraints_string);
  boa.accuracy_test(soln.hypas);

  
}



int main()
{
  size_t m     = 55;
  size_t k     = 118;
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
            std::cout << "\ntest " << testi << "/32";
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
