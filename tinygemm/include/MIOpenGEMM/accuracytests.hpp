#ifndef ACCURACYTESTS_HPP
#define ACCURACYTESTS_HPP


#include <algorithm>
#include <sstream>

#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/outputwriter.hpp>

namespace MIOpenGEMM {
namespace accuracytests {

template <typename TFloat>
void elementwise_compare(const TFloat * c_before, double beta, const TFloat * c_cpu, const TFloat * c_gpu, unsigned nels, outputwriting::OutputWriter & mowri);


}
} //namespaces

#endif
