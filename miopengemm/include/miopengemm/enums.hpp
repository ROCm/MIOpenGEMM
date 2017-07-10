/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ALLENUMS_HPP
#define GUARD_MIOPENGEMM_ALLENUMS_HPP

#include <vector>
#include <unordered_map>
#include <miopengemm/error.hpp>


namespace MIOpenGEMM {


template <typename T>
class EnumMapper{
  public:
    size_t n;
    std::vector<T> name;
    std::unordered_map<T, size_t> val;
    EnumMapper(const std::vector<T> & name_); 
};

// if you add a parameter to enum, make sure to add it before the final count N

namespace BasicKernelType
{
enum E{
  WSA = 0,
  WSB,
  BETAC,
  MAIN,
  N  // how many BasicKernelTypes
};
extern const EnumMapper<std::string> M;

}


namespace SummStat
{
enum E{
  MEAN = 0,
  MEDIAN,
  MAX,
  N
};
extern const EnumMapper<std::string> M;

}

namespace Chi
{
enum E{
  MIC = 0,
  PAD,
  PLU,
  LIW,
  MIW,
  WOS,
  N
};
extern const EnumMapper<std::string> M;
}


namespace NonChi
{
enum E{
  UNR = 0,
  GAL,
  PUN,
  ICE,
  NAW,
  UFO,
  MAC,
  SKW,
  N
};
extern const EnumMapper<std::string> M;
}



namespace Mat
{
enum E{
  A = 0,
  B,
  C,
  N
};
extern const EnumMapper<char> M;
}


namespace Mem
{
enum E{
  A,
  B,
  C,
  W,
  N
};
extern const EnumMapper<char> M;
}




namespace Status
{
enum E{
  UNDEFINED = -1
};
}

namespace Binary
{
enum E{
  NO  = 0,
  YES = 1
};
}

namespace GroupAllocation
{
enum E{
  BYROW = 1,
  BYCOL = 2,
  SUCOL = 3
};
}

namespace Scratch
{
enum E{
  UNUSED = 0,
  COPY,
  NFORM 
};
}




}

#endif
