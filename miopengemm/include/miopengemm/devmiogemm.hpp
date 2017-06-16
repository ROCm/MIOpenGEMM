#ifndef DEGEMMAPIQQ_HPP
#define DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <vector>
#include <string>

namespace MIOpenGEMM{
namespace dev{

template <typename TFloat>
void benchgemm(const std::vector<std::string> & hyperstrings,         
unsigned n_runs, const Geometry & gg, const Offsets & toff,  const TFloat * a, const TFloat * b, const TFloat * c, outputwriting::OutputWriter & mowri);


template <typename TFloat>
void accuracy_test(const std::string & hyperstring, 
const Geometry & gg, const Offsets & toff, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test, outputwriting::OutputWriter & mowri);


template <typename TFloat>
Solution find(const FindParams & find_params,  const TFloat * a, const TFloat * b, const TFloat * c, std::string constraints_string, const Geometry & gg, const Offsets & toff, outputwriting::OutputWriter & mowri);

}
}

#endif
