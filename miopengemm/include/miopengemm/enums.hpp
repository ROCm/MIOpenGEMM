/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ALLENUMS_HPP
#define GUARD_MIOPENGEMM_ALLENUMS_HPP


namespace MIOpenGEMM {

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
}


namespace SummStat
{
enum E{
  MEAN = 0,
  MEDIAN,
  MAX,
  N
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


namespace Mat
{
enum E{
  A,
  B,
  C,
  N
};
}
}

#endif
