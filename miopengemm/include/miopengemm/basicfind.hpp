#ifndef BASICFIND_HPP
#define BASICFIND_HPP

#include <miopengemm/solution.hpp>
namespace MIOpenGEMM{
  
Solution basicfind(const Geometry & geometry, const Offsets & toff, const FindParams & find_params, bool verbose, std::string logfile, std::string constraints_string, unsigned n_postfind_runs, bool do_cpu_test);

}

#endif

