/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/tinytwo.hpp>

template <typename TFloat>
void geometrytest(const MIOpenGEMM::Geometry& gg)
{
  using namespace MIOpenGEMM;

  // CLHint         devhint;
  CLHint         devhint(0, 0);                    //(first platform, first device)
  Offsets        offsets = get_padding_offsets();  // get_zero_offsets();
  owrite::Writer mowri(Ver::E::TERMINAL, "");
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);

  // FindParams find_params = get_quick_find_params();
  FindParams  find_params = get_at_least_n_seconds(.05);  //(1, 1.14, 2, 200., SummStat::E::MAX);
  std::string constraints_string = "";                    // A_WOS0__B_WOS0__C_UFO1_ICE1_IWI0";
  Solution    soln               = boa.find2(find_params, constraints_string);
  std::cout << '\n' << soln.hypas.get_string() << '\n';
  boa.accuracy_test(soln.hypas);
}

int main()
{
  using namespace MIOpenGEMM;

  size_t m              = 81;
  size_t k              = 57;
  size_t testi          = 0;
  size_t workspace_size = 1000 * 1000;

  for (bool tC : {false})  // true not necessary.
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
            std::cout << "\n\n\ntest " << testi << "/32\n";
            Geometry gg =
              get_padded_geometry<float>(isColMajor, tA, tB, tC, m, n, k, workspace_size);
            std::cout << "<float>  " << gg.get_string() << '\n';
            geometrytest<float>(gg);

            bool dodoub = false;
            if (dodoub)
            {
              gg = get_padded_geometry<double>(isColMajor, tA, tB, tC, m, n, k, workspace_size);
              std::cout << "\n\n<double>  " << gg.get_string() << '\n';
              geometrytest<double>(gg);
            }
          }
        }
      }
    }
  }

  std::cout << "\n\n";
  return 0;
}
