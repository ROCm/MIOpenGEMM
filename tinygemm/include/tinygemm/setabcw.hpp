#ifndef SETABCW_HPP
#define SETABCW_HPP

#include <tinygemm/geometry.hpp>


namespace setabcw{


template <typename TFloat>
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);

template <typename TFloat>     
void set_abcw(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, std::vector<TFloat> & v_workspace, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff);


}

#endif
