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

Bundle get_bundle(const hyperparams::HyperParams& hp,
                  const Geometry&                 gg,
                  outputwriting::OutputWriter&    mowri,
                  bool                            bundle_verbose)
{

  derivedparams::DerivedParams dp(hp, gg);

  std::vector<KernelString>          v_tgks;
  std::vector<std::vector<size_t>> v_wait_indices;


  for (auto emat_x : {Mat::E::A, Mat::E::B}){

    if (hp.at(emat_x).vs[Chi::E::WOS] == Scratch::E::UNUSED)
    {
      // no workspace kernel
    }
  
    else if (hp.at(emat_x).vs[Chi::E::WOS] == Scratch::E::COPY)
    {
      v_tgks.emplace_back(copygen::get_copy_kernelstring(emat_x, hp, gg, dp));
    }
  
    else if (hp.at(emat_x).vs[Chi::E::WOS] == Scratch::E::NFORM)
    {
      v_tgks.emplace_back(nformgen::get_nform_kernelstring(emat_x, hp, gg, dp));
    }
  
    else
    {
      throw miog_error("hp.at(emat_x).vs[Chi::E::WOS] should be 0, 1 or 2 (Scratch::E::UNUSED , Scratch::E::COPY or Scratch::E::NFORM)");
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

  std::vector<KernelType> types;
  for (size_t i = 0; i < v_tgks.size(); ++i)
  {
    types.push_back(v_tgks[i].type);
  }

  for (size_t i = 0; i < v_tgks.size(); ++i)
  {
    v_wait_indices.push_back({});
    for (size_t j = 0; j < v_tgks.size(); ++j)
    {
      if (std::find(kernel_dependencies.at(types[i].basic_kernel_type).begin(),
                    kernel_dependencies.at(types[i].basic_kernel_type).end(),
                    types[j].basic_kernel_type) !=
          kernel_dependencies.at(types[i].basic_kernel_type).end())
      {
        v_wait_indices.back().push_back(j);
      }
    }
  }

  if (bundle_verbose == true)
  {
    mowri << "\n";
    mowri << "network of kernel dependencies: \n";
    for (size_t i = 0; i < v_tgks.size(); ++i)
    {
      std::stringstream pre_waits_for_ss;
      pre_waits_for_ss << "kernel " << i << " ( " << types[i].full << " )";
      std::string pre_waits_for = pre_waits_for_ss.str();
      mowri << pre_waits_for;
      int         base_space(26);
      std::string space1(std::max(1, base_space - static_cast<int>(pre_waits_for.size())), ' ');
      mowri << space1 << "waits for   " << Flush;

      if (v_wait_indices[i].size() == 0)
      {
        mowri << "(nothing)";
      }

      for (size_t j = 0; j < v_wait_indices[i].size(); ++j)
      {
        mowri << "(kernel " << v_wait_indices[i][j] << " ( " << types[v_wait_indices[i][j]].full
              << " ))   " << Flush;
      }
      mowri << Endl;
    }
    mowri << "\n";
  }

  return Bundle(std::move(v_tgks), std::move(v_wait_indices), std::move(dp));
}
}
}
