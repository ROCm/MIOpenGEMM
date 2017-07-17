/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <miopengemm/alphagenerator.hpp>
#include <miopengemm/betacgenerator.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/copygenerator.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/normalformgenerator.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{
namespace kerngen
{

Bundle::Bundle(const HyPas& hp_, const Geometry& gg_, owrite::Writer& mowri):hp(hp_), gg(gg_), dp(hp, gg)
{


  //std::vector<KernBlob>  v_tgks;
  //std::vector<std::vector<size_t>>v_wait_indices;

  for (auto emat_x : {Mat::E::A, Mat::E::B})
  {

    if (hp.sus[emat_x].vs[Chi::E::WOS] == Scratch::E::UNUSED)
    {
      // no workspace kernel
    }

    else if (hp.sus[emat_x].vs[Chi::E::WOS] == Scratch::E::COPY)
    {
      v_tgks.emplace_back(copygen::get_copy_kernelstring(emat_x, hp, gg, dp));
    }

    else if (hp.sus[emat_x].vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      v_tgks.emplace_back(nformgen::get_nform_kernelstring(emat_x, hp, gg, dp));
    }

    else
    {
      std::stringstream errm;
      errm << "hp.sus[emat_x].vs[Chi::E::WOS] should be 0, 1 or 2"
           << "(Scratch::E::UNUSED , Scratch::E::COPY or Scratch::E::NFORM)";
      throw miog_error(errm.str());
    }
  }

  if (dp.main_does_beta_c_inc == 0)
  {
    v_tgks.emplace_back(betacgen::get_betac_kernelstring(hp, gg, dp));
  }

  v_tgks.emplace_back(alphagen::get_alpha_kernelstring(hp, gg, dp));

  // indent the kernel strings, in case someone wants to
  // print them. For (v-minorly) better
  // performance, this should not be done
  /* TODO : can clang-format be used for this ? */
  for (auto& x : v_tgks)
  {
    stringutil::indentify(x.kernstr);
  }

  for (size_t i = 0; i < v_tgks.size(); ++i)
  {
   v_wait_indices.push_back({});
    for (size_t j = 0; j < v_tgks.size(); ++j)
    {
      if (std::find(KType::dependencies.at(v_tgks[i].e_ktype).begin(),
                    KType::dependencies.at(v_tgks[i].e_ktype).end(),
                    v_tgks[j].e_ktype) !=
          KType::dependencies.at(v_tgks[i].e_ktype).end())
      {
       v_wait_indices.back().push_back(j);
      }
    }
  }
    
  mowri.bw[OutPart::E::DEP] << "\nnetwork of kernel dependencies: \n";
  for (size_t i = 0; i < v_tgks.size(); ++i)
  {
    std::stringstream ss1;
    ss1 << "kernel " << i << " {" << v_tgks[i].kuses.full << "}";
    std::string pre_waits_for = ss1.str(); 
    
    if (pre_waits_for.size() < 35){
      pre_waits_for.resize(37, ' ');
    }
    mowri.bw[OutPart::E::DEP] << pre_waits_for << " waits for :  " << Flush;
    if (v_wait_indices[i].size() == 0)
    {
      mowri.bw[OutPart::E::DEP] << "nothing";
    }

    for (size_t j = 0; j <v_wait_indices[i].size(); ++j)
    {
      mowri.bw[OutPart::E::DEP] << v_wait_indices[i][j]  << '{' << v_tgks[v_wait_indices[i][j]].kuses.full  << "} " << Flush;
    }
    mowri.bw[OutPart::E::DEP] << Endl;
  }
  mowri.bw[OutPart::E::DEP] << '\n';

  //return Bundle(std::move(v_tgks), std::move(wait_indices), std::move(dp));
}
}
}
