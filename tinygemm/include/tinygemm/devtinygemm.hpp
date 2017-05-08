#ifndef DEGEMMAPIQQ_HPP
#define DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <vector>
#include <string>

namespace tinygemm{
namespace dev{

template <typename TFloat>
//void benchgemm(const std::vector<hyperparams::HyperParams> & hps,         
void benchgemm(const std::vector<std::string> & hyperstrings,         
unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff,  const TFloat * a, const TFloat * b, const TFloat * c, outputwriting::OutputWriter & mowri);


template <typename TFloat>
//void accuracy_test(const hyperparams::HyperParams & hp, 
void accuracy_test(const std::string & hyperstring, 
const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test, outputwriting::OutputWriter & mowri);


template <typename TFloat>
tinygemm::TinyGemmSolution find(float allotted_time, const TFloat * a, const TFloat * b, const TFloat * c, std::string constraint_string, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff, outputwriting::OutputWriter & mowri);

}
}

#endif
