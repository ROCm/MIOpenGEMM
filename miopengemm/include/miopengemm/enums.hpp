/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ALLENUMS_HPP
#define GUARD_MIOPENGEMM_ALLENUMS_HPP

#include <array>
#include <unordered_map>
#include <vector>
#include <miopengemm/error.hpp>

namespace MIOpenGEMM
{

namespace Floating
{

const double& get_default_alpha();
const double& get_default_beta();

class MFType
{
  private:
  double v_d;
  float  v_f;

  public:
  MFType(double v);
  const void* operator[](char floattype) const;
};

const MFType& get_m_alpha();

const MFType& get_m_beta();
}

template <typename T>
class EnumMapper
{
  public:
  size_t              N;
  std::vector<T>      name;
  std::vector<T>      lcase_name;
  std::vector<size_t> all_enum;
  std::unordered_map<T, size_t> val;
  EnumMapper(const std::vector<T>& name_);
};

// if you add a parameter to enum, make sure to add it before the final count N

namespace SummStat
{
enum E
{
  MEAN = 0,
  MEDIAN,
  MAX,
  N
};
const EnumMapper<std::string>& M();
}

namespace Xtr
{
enum E
{
  MIN = 0,
  MAX,
  N
};
const EnumMapper<std::string>& M();
}

namespace Chi
{
enum E
{
  MIC = 0,
  PAD,
  PLU,
  LIW,
  MIW,
  WOS,
  VEW,  // vector width
  N
};
const EnumMapper<std::string>& M();

// prior weight on importance in graph
const std::vector<int>& get_priority();
}

namespace NonChi
{
enum E
{
  UNR = 0,  // tile to load, dimension in k (unroll)
  GAL,      // group allocation
  PUN,      // use pragma unroll indiscriminately
  ICE,      // split work group in the k-dimension as well
  IWI,      // (if ICE != 1) inter-weave the work in the k-dimension
  SZT,      // use size_t (ulong) in the kernels instead of unsigned/short
  MAD,      // use mad instead of c += a*b;
  NAW,      // (if GAL == 3) number of work-groups per meta-tile
  UFO,      // start work at some work-group dependent starting back offset
  MAC,      // work items per work group
  SKW,      // skewness of work-item grid of work group
  AFI,      // do A loops and defs first. outerloops over a dimensions.
  MIA,      // work item allocation within workgroup : % or /
  N
};
const EnumMapper<std::string>& M();

const std::vector<int>& get_priority();
}

namespace Mat
{
enum E
{
  A = 0,
  B,
  C,
  N
};
const EnumMapper<char>& M();
}

namespace Mem
{
enum E
{
  A = 0,
  B,
  C,
  W,
  N
};
const EnumMapper<char>& M();

Mem::E mat_to_mem(Mat::E);
}

namespace Mat
{
Mat::E                         mem_to_mat(Mem::E);
const EnumMapper<std::string>* mat_to_xchi(Mat::E);
const std::vector<int>*        mat_to_priority(Mat::E);
}

namespace Status
{
enum E
{
  UNDEFINED = -1
};
}

namespace Binary
{
enum E
{
  NO  = 0,
  YES = 1
};
}

namespace IfNoCache
{
enum E
{
  GENERIC = 0,
  RANDOM  = 1
};
}

namespace GroupAllocation
{
enum E
{
  BYROW = 1,
  BYCOL = 2,
  SUCOL = 3
};
}

namespace MicroAllocation
{
enum E
{
  BYA = 0,
  BYB = 1,
  N
};
const EnumMapper<std::string>& M();
}

namespace Scratch
{
enum E
{
  UNUSED = 0,
  COPY,
  NFORM
};
}

namespace OutPart
{
enum E
{
  MAI = 0,
  TRA,
  DEP,
  ACC,
  WRN,
  CCH,
  BEN,
  MER,
  N
};
const EnumMapper<std::string>& M();
}

namespace Ver
{
enum E
{
  SILENT = 0,    // no output anywhere, absolute silence
  TERMINAL,      // all output to terminal, other than tracker
  SPLIT,         // main output to terminal and file
  TOFILE,        // main output to file
  TRACK,         // tracker output to terminal
  STRACK,        // tracker output to terminal, main output to file
  ACCURACY,      // tracker and accuracy to terminal
  TERMWITHDEPS,  // like terminal, with additionally the dependencies between kernels
  MERGE,         // when merging two kernel caches
  MULTIBENCH,
  N
};
const EnumMapper<std::string>& M();

const std::array<bool, E::N>& get_fileRequired();
const std::array<std::array<bool, OutPart::E::N>, E::N>& get_toFile();
const std::array<std::array<bool, OutPart::E::N>, E::N>& get_toTerm();
}

namespace KType
{
enum E
{
  WSA = 0,
  WSB,
  BETAC,
  MAIN,
  N  // how many KTypes
};
const EnumMapper<std::string>& M();

// maps the dependencices of kernels, order of execution
// For example deps[MAIN] = {WSA, WSB, BETAC},
// as all of these must first complete
// before MAIN can execute
const std::array<std::vector<size_t>, KType::N>& get_dependencies();
}
}

#endif
