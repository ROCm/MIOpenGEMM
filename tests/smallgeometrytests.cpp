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
  size_t  workspace_size   = 3;
  Geometry gg = get_padded_geometry<TFloat>(isColMajor, tA, tB, tC, m, n, k, workspace_size); 
  FindParams find_params = get_quick_find_params();
  
  
  //create a mowri!
  outputwriting::OutputWriter mowri(Ver::E::TRACK, "");

  
  std::string constraints_string = "A_WOS0__B_WOS0__C_ICE3";

  Solution soln = dev::basicfind(
    find_params, constraints_string, gg, offsets, mowri);
  
  //dev::accuracy_test(
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
