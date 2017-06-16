#ifndef SETABCW_HPP
#define SETABCW_HPP

#include <tinygemm/geometry.hpp>


namespace tinygemm{
namespace setabcw{


template <typename TFloat>
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const TinyGemmGeometry & gg, const TinyGemmOffsets & toff);

template <typename TFloat>     
void set_abcw(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, std::vector<TFloat> & v_workspace, const TinyGemmGeometry & gg, const TinyGemmOffsets & toff);


}
}

#endif
