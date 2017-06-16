#ifndef ARCHITESTS_HPP
#define ARCHITESTS_HPP


#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/derivedparams.hpp>
#include <CL/cl.h>

namespace MIOpenGEMM{
namespace architests{
  
  std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue, const hyperparams::HyperParams &, const derivedparams::DerivedParams &);

}
}

#endif
