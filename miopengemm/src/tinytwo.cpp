/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <miopengemm/accuracytests.hpp>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/floattostring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/tinytwo.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{
namespace dev
{

TinyTwo::TinyTwo(Geometry gg_, Offsets toff_, owrite::Writer& mowri_, const CLHint& xhint)
{

  switch (gg_.floattype)
  {

  case 'f': f_moa.reset(new TinyOne<float>(gg_, toff_, mowri_, xhint)); break;
  case 'd': d_moa.reset(new TinyOne<double>(gg_, toff_, mowri_, xhint)); break;
  default: throw miog_error("unrecognised floattype char in TinyTwo constructor");
  }

  active_type = gg_.floattype;
}

std::vector<std::vector<double>> TinyTwo::benchgemm(const std::vector<HyPas>& hps, const Halt& hl)
{
  switch (active_type)
  {
  case 'f': return f_moa->benchgemm(hps, hl);
  case 'd': return d_moa->benchgemm(hps, hl);
  default: throw miog_error("unrecognised floattype char in TinyTwo benchgemm");
  }
}

void TinyTwo::accuracy_test(const HyPas& hp)
{

  switch (active_type)
  {
  case 'f': f_moa->accuracy_test(hp); break;
  case 'd': d_moa->accuracy_test(hp); break;
  default: throw miog_error("unrecognised floattype char in TinyTwo accuracy_test with 1 parm");
  }
}

Solution TinyTwo::find2(const FindParams& find_params, const Constraints& constraints)
{
  switch (active_type)
  {
  case 'f': return f_moa->find1(find_params, constraints);
  case 'd': return d_moa->find1(find_params, constraints);
  default: throw miog_error("unrecognised floattype char in TinyTwo find");
  }
}

template <>
std::unique_ptr<TinyOne<float>>& TinyTwo::get_up_moa<float>()
{
  return f_moa;
}

template <>
std::unique_ptr<TinyOne<double>>& TinyTwo::get_up_moa<double>()
{
  return d_moa;
}

template <>
void TinyTwo::set_active_type<float>()
{
  active_type = 'f';
}

template <>
void TinyTwo::set_active_type<double>()
{
  active_type = 'd';
}
}
}
