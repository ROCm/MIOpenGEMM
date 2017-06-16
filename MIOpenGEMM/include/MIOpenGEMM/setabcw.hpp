#ifndef SETABCW_HPP
#define SETABCW_HPP

#include <MIOpenGEMM/geometry.hpp>


namespace MIOpenGEMM{
namespace setabcw{


template <typename TFloat>
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const Geometry & gg, const Offsets & toff);

template <typename TFloat>     
void set_abcw(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, std::vector<TFloat> & v_workspace, const Geometry & gg, const Offsets & toff);


}
}

#endif
