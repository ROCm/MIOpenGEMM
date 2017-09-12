/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ERROR_HPP
#define GUARD_MIOPENGEMM_ERROR_HPP

#include <stdexcept>

namespace MIOpenGEMM
{

class miog_error : public std::runtime_error
{
  public:
  miog_error(const std::string& what_arg);
};
}

#endif
