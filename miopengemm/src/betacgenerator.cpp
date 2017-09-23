/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <miopengemm/betacgenerator.hpp>

namespace MIOpenGEMM
{
namespace betacgen
{

BetacGenerator::BetacGenerator(const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_)

  : bylinegen::ByLineGenerator(Mat::E::C, hp_, gg_, dp_)
{
}

void BetacGenerator::set_type() { type = "betac"; }

size_t BetacGenerator::get_local_work_size() { return dp.betac_local_work_size; }

size_t BetacGenerator::get_work_per_thread() { return dp.betac_work_per_thread; }

KType::E BetacGenerator::get_ktype() { return KType::E::BETAC; }

void BetacGenerator::setup_additional()
{
  description_string = R"(
/* ****************************************************
* It is used to perform the beta*C step in GEMM, 
* where recall GEMM has C <- alpha*A*B + beta*C
* It is not quite an axpy, as when ldc is not minimal, 
* C is not contiguous memory  
****************************************************** */ )";
  // inner_work_string  = "\n/* the beta scaling */\nc[i] *= beta;";
  inner_work_string =
    "\n/* beta scaling */\nif (beta <= 0 && beta >= 0){c[i] = 0;}else{c[i] *= beta;}";
}

void BetacGenerator::append_derived_definitions_additional(std::stringstream& ss) { ss << " "; }

KernBlob get_betac_kernelstring(const HyPas& hp, const Geometry& gg, const DerivedParams& dp)
{
  BetacGenerator bcg(hp, gg, dp);
  bcg.setup();
  return bcg.get_kernelstring();
}
}
}
