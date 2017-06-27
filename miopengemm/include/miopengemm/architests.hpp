
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef ARCHITESTS_HPP
#define ARCHITESTS_HPP


#include <miopengemm/hyperparams.hpp>
#include <miopengemm/derivedparams.hpp>
#include <CL/cl.h>

namespace MIOpenGEMM{
namespace architests{
  
  std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue, const derivedparams::DerivedParams &, const Geometry & gg, const hyperparams::HyperParams &);

}
}

#endif
