#ifndef TINYGEMM_DERIVEDPARAMS_HPP
#define TINYGEMM_DERIVEDPARAMS_HPP

#include <vector>
#include <string>
#include <map>
#include <limits>

namespace tinygemm{
namespace derivedparams{

class DerivedParams{
  
  private: 
    unsigned uninitialised_unsigned = std::numeric_limits<unsigned>::max();
    
  
  public:


    /* TODO : write descriptions */

    
    
    unsigned split_on_k = uninitialised_unsigned;
    unsigned does_alpha_c_inc = uninitialised_unsigned;
    unsigned macro_tile_area = uninitialised_unsigned;
    unsigned micro_tile_area = uninitialised_unsigned;
    unsigned n_workitems_per_workgroup = uninitialised_unsigned;
    unsigned macro_tile_height_and_pad = uninitialised_unsigned;
    unsigned macro_tile_width_and_pad = uninitialised_unsigned;
    unsigned n_elements_in_a_unroll = uninitialised_unsigned;
    unsigned n_elements_in_b_unroll = uninitialised_unsigned;
    unsigned n_elements_in_padded_a_unroll = uninitialised_unsigned;
    unsigned n_elements_in_padded_b_unroll = uninitialised_unsigned;
    unsigned n_elements_of_a_to_load_per_workitem = uninitialised_unsigned;
    unsigned n_elements_of_b_to_load_per_workitem = uninitialised_unsigned;
    unsigned n_micro_tiles_vertically = uninitialised_unsigned;
    unsigned n_micro_tiles_horizontally = uninitialised_unsigned;
    unsigned micro_a_tile_pll_unroll = uninitialised_unsigned; 
    unsigned micro_a_tile_perp_unroll = uninitialised_unsigned; 
    unsigned micro_b_tile_pll_unroll = uninitialised_unsigned;
    unsigned micro_b_tile_perp_unroll = uninitialised_unsigned;    
    unsigned n_micro_a_tiles_pll_unroll = uninitialised_unsigned;  
    unsigned n_micro_b_tiles_pll_unroll = uninitialised_unsigned;
  

    /* */
    std::string strided_i_vertical;
    /* */
    std::string strided_i_horizontal; 
    /*the int type for atomics */
    std::string infa;
    /* the function to use for atomic ints */
    std::string fati;
    /* */
    std::string effective_k; 
    /* */
    std::string alpha_scaled;
    /* pragma unroll string ("#pragma unroll\n" or "") */
    std::string pragma_unroll_string;
    /* currently one of "float" and "double", set from float_size */
    std::string t_float;
    
    
    
};

}
}


#endif
