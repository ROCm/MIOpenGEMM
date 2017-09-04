/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <string>
#include <miopengemm/geometries.hpp>
#include <miopengemm/tinytwo.hpp>

int main(int argc, char* argv[])
{

  using namespace MIOpenGEMM;
  CLHint                   devhint;
  Offsets                  offsets = get_zero_offsets();
  std::vector<std::string> sargs;
  for (size_t i = 1; i < argc; ++i)
  {
    sargs.push_back(argv[i]);
  }

  if (sargs.size() != 2 && sargs.size() != 3)
  {
    throw miog_error(
      "should be 2/3 arguments in multifindbase : basedir gg_string (constraints_string)");
  }

  std::string basedir(sargs[0]);
  Geometry    gg(sargs[1]);
  Constraints cons("");
  if (sargs.size() == 3)
  {
    cons = sargs[2];
  }

  auto dirname = basedir + "gg_" + gg.get_string() + "/cns_" + cons.get_r_str() + "/";
  // WARNING : mkdir only works on linux/mac
  std::string syscall = "mkdir -p " + dirname;
  std::system(syscall.c_str());
  auto        fn             = dirname + "log.txt";
  std::string fn_final       = basedir + "cacheentries.txt";
  std::string fn_final_local = dirname + "cacheentry.txt";

  std::cout << fn;
  std::cout << '\n';

  owrite::Writer mowri(Ver::E::STRACK, fn);
  dev::TinyTwo   boa(gg, offsets, mowri, devhint);
  auto           find_params = get_at_least_n_restarts(5);
  find_params.sumstat        = SummStat::E::MEDIAN;
  auto soln                  = boa.find2(find_params, cons);

  std::cout << soln.hypas.get_string() << "   :   " << gg.get_gflops(soln.extime) << " gflops ";
  std::cout << '\n' << '\n';
  owrite::Writer mowri_final_local(Ver::E::TOFILE, fn_final_local);
  mowri_final_local << soln.get_cache_entry_string() << "\n";
  std::ofstream outfile;
  outfile.open(fn_final, std::ios_base::app);
  outfile << soln.get_cache_entry_string() << "\n\n";
  return 0;
}
