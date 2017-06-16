#ifndef ARCHITESTS_HPP
#define ARCHITESTS_HPP


#include <miopengemm/hyperparams.hpp>
#include <miopengemm/derivedparams.hpp>
#include <CL/cl.h>

namespace MOOMOOMOOGEMM{
namespace architests{
  
  std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue, const hyperparams::HyperParams &, const derivedparams::DerivedParams &);

}
}

#endif
