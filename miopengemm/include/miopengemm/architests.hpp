#ifndef GUARD_MIOPENGEMM_ARCHITESTS_HPP
#define GUARD_MIOPENGEMM_ARCHITESTS_HPP

#include <CL/cl.h>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/hyperparams.hpp>

namespace MIOpenGEMM
{
namespace architests
{

std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue,
                                                          const derivedparams::DerivedParams&,
                                                          const Geometry& gg,
                                                          const hyperparams::HyperParams&);
}
}

#endif
