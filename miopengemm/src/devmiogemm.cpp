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
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/floattostring.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/slowcpugemm.hpp>
#include <miopengemm/stringutilbase.hpp>


namespace MIOpenGEMM
{
namespace dev
{

Boa::Boa(Geometry gg_, Offsets toff_, owrite::Writer& mowri_)
{
  
  switch (gg_.floattype)
  {
  case 'f': f_moa.reset(new Diva<float>(gg_, toff_, mowri_)); break;
  case 'd': d_moa.reset(new Diva<double>(gg_, toff_, mowri_)); break;
  default: throw miog_error("unrecognised floattype char in Boa constructor");
  }
  active_type = gg_.floattype;
}

void Boa::benchgemm(const std::vector<std::string>& hyperstrings,
                    size_t                          max_number_of_runs,
                    double                          max_time_per_kernel)
{
  switch (active_type)
  {
  case 'f': f_moa->benchgemm(hyperstrings, max_number_of_runs, max_time_per_kernel); break;
  case 'd': d_moa->benchgemm(hyperstrings, max_number_of_runs, max_time_per_kernel); break;
  default: throw miog_error("unrecognised floattype char in Boa benchgemm");
  }
}

void Boa::accuracy_test(const std::string& hyperstring){

  switch (active_type)
  {
  case 'f': f_moa->accuracy_test(hyperstring, nullptr); break;
  case 'd': d_moa->accuracy_test(hyperstring, nullptr); break;
  default: throw miog_error("unrecognised floattype char in Boa accuracy_test with 1 parm");
  }  
}
  
Solution Boa::find(const FindParams& find_params, std::string constraints_string)
{
  switch (active_type)
  {
  case 'f': return f_moa->find(find_params, constraints_string);
  case 'd': return d_moa->find(find_params, constraints_string);
  default: throw miog_error("unrecognised floattype char in Boa find");
  }
}

template <>
std::unique_ptr<Diva<float>>& Boa::get_up_moa<float>()
{
  return f_moa;
}

template <>
std::unique_ptr<Diva<double>>& Boa::get_up_moa<double>()
{
  return d_moa;
}

template <>
void Boa::set_active_type<float>()
{
  active_type = 'f';
}

template <>
void Boa::set_active_type<double>()
{
  active_type = 'd';
}

}
}

