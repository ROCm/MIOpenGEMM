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

static const double default_alpha = 0.415693029182345929;
static const double default_beta  = 0.273539340934809345;

class MFType
{
  private:
  double v_d;
  float  v_f;

  public:
  MFType(double v);
  void* operator[](char floattype) const;
};

static const MFType m_alpha(default_alpha);
static const MFType m_beta(default_beta);
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
extern const EnumMapper<std::string> M;
}

namespace Xtr
{
enum E
{
  MIN = 0,
  MAX,
  N
};
extern const EnumMapper<std::string> M;
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
extern const EnumMapper<std::string> M;

// prior weight on importance in graph
extern const std::vector<int> priority;
}

namespace NonChi
{
enum E
{
  UNR = 0,
  GAL,
  PUN,
  ICE,
  IWI,
  SZT,
  NAW,
  UFO,
  MAC,
  SKW,
  AFI,  // a loops and defs first. outerloops over a dimensions.
  MIA,  // micro allocation
  N
};
extern const EnumMapper<std::string> M;
extern const std::vector<int>        priority;
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
extern const EnumMapper<char> M;
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
extern const EnumMapper<char> M;
Mem::E                        mat_to_mem(Mat::E);
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
  GENERIC= 0,
  RANDOM = 1
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
extern const EnumMapper<std::string> M;
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
extern const EnumMapper<std::string> M;
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
  MULTIBENCH,
  MERGE,  // when merging two kernel caches
  N
};
extern const EnumMapper<std::string> M;
extern const std::array<bool, E::N> fileRequired;
extern const std::array<std::array<bool, OutPart::E::N>, E::N> toFile;
extern const std::array<std::array<bool, OutPart::E::N>, E::N> toTerm;
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
extern const EnumMapper<std::string> M;

// maps the depencices of kernels, order of exection
// For example deps[MAIN] = {WSA, WSB, BETAC},
// as all of these must first complete
// before MAIN can execute
extern std::array<std::vector<size_t>, E::N> dependencies;
}
}

#endif
