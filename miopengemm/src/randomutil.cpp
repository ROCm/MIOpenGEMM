/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <miopengemm/randomutil.hpp>

namespace MIOpenGEMM
{

RandomUtil::RandomUtil() : rd(), gen(rd()) {}

size_t RandomUtil::get_from_range(size_t upper) { return unidis(gen) % upper; }
}
