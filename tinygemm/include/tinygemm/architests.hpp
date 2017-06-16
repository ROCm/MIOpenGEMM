#ifndef ARCHITESTS_HPP
#define ARCHITESTS_HPP


#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>
#include <CL/cl.h>

namespace tinygemm{
namespace architests{
  
  std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue, const hyperparams::HyperParams &, const derivedparams::DerivedParams &);

}
}

#endif
