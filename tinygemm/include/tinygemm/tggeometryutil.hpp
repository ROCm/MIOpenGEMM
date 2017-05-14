#ifndef TG_PROBLEMGEOMETRY_UTIL_HPP
#define TG_PROBLEMGEOMETRY_UTIL_HPP

#include <tinygemm/tinygemmgeometry.hpp>

#include <string>
#include <vector>

namespace tinygemm{


std::vector<tinygemm::TinyGemmGeometry> get_from_m_n_k_ldaABC_tA_tB(const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> &  basicgeos, unsigned workspace_size);

std::vector<tinygemm::TinyGemmGeometry> get_from_m_n_k_tA_tB(const std::vector<std::tuple<unsigned, unsigned, unsigned, bool, bool>> &  basicgeos, unsigned workspace_size );


}

#endif
