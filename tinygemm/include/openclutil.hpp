#ifndef OPENCLUTIL_H
#define OPENCLUTIL_H


#include "outputwriter.hpp"
#include <CL/cl.h>

namespace tinygemm{
namespace openclutil{

void set_platform_etc(cl_platform_id & platform, cl_uint & num_platforms, cl_context & context, cl_device_id & device_id_to_use, outputwriting::OutputWriter & mowri);

void set_program_and_kernel(cl_program & program, cl_kernel & kernel, std::string & kernel_function_name, const cl_context & context, const cl_device_id & device_id_to_use, const std::string & filename);  
  
} // end namesapce
}

#endif
