#ifndef DEGEMMAPIQQ_HPP
#define DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <vector>
#include <string>
namespace tinygemm{
namespace dev{


template <typename TFloat>
void benchgemm(const std::vector<hyperparams::HyperParams> & hps,         
unsigned n_runs, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b, const TFloat * c, bool verbose = true, std::string logfile = "");

template <typename TFloat>
void accuracy_test(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, const TFloat * a, const TFloat * b,
const TFloat * c, const TFloat * c_true_for_test = nullptr, bool verbose = true, std::string logfile = "");

template <typename TFloat>
tinygemm::TinyGemmSolution find(float allotted_time, const TFloat * a, const TFloat * b, const TFloat * c, bool enforce_deterministic, const tinygemm::TinyGemmGeometry & gg, const double alpha, const double beta, bool verbose, std::string logfile);

}
}

#endif
