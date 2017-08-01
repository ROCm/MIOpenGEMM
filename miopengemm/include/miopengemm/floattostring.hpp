/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_FLOATTOSTRING_HPP
#define GUARD_MIOPENGEMM_FLOATTOSTRING_HPP

#include <string>

namespace MIOpenGEMM
{
namespace floattostring
{

std::string float_string_type(double x);

std::string float_string_type(float x);

char float_char_type(double x);

char float_char_type(float x);

template <typename TFloat>
char get_float_char()
{
  TFloat x = 0.;
  return float_char_type(x);
}

template <typename TFloat>
std::string get_float_string()
{
  TFloat x = 0.;
  return float_string_type(x);
}

std::string get_float_string(char floattype);
}
}
#endif
