
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef ACCURACYTESTS_HPP
#define ACCURACYTESTS_HPP


#include <algorithm>
#include <sstream>

#include <miopengemm/error.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM {
namespace accuracytests {

template <typename TFloat>
void elementwise_compare(const TFloat * c_before, double beta, const TFloat * c_cpu, const TFloat * c_gpu, unsigned nels, outputwriting::OutputWriter & mowri);


}
} //namespaces

#endif
