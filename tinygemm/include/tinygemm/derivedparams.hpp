#ifndef TINYGEMM_DERIVEDPARAMS_HPP
#define TINYGEMM_DERIVEDPARAMS_HPP

#include <vector>
#include <string>
#include <map>
#include <limits>
#include <tuple>

#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>

namespace tinygemm{
namespace derivedparams{

const unsigned uninitialised_unsigned = std::numeric_limits<unsigned>::max();

std::tuple<bool, std::string>
get_deriveability(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);

/* GA 3 specific derived parameters */
class GA3Params{
public:
  unsigned super_column_width = uninitialised_unsigned;
  unsigned last_super_column_width = uninitialised_unsigned;
};


/* all derived parameters */
class DerivedParams{

public:
  
  /* initiate all parameters, throwing an error if there is an incompatibility */
  DerivedParams(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  
  DerivedParams(std::string s);
  
  /* does the minimum setting to confirm compatibitily. called by get_deriveability */
  std::tuple<bool, std::string> set_fragile(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  
  /* TODO : write descriptions */
  unsigned macro_tile_area = uninitialised_unsigned;
  unsigned micro_tile_area = uninitialised_unsigned;
  //TODO : rename to reflect alpha kernel
  unsigned n_work_items_per_workgroup = uninitialised_unsigned;
  unsigned n_elements_in_a_unroll = uninitialised_unsigned;
  unsigned n_elements_in_b_unroll = uninitialised_unsigned;
  unsigned n_elements_of_a_to_load_per_workitem = uninitialised_unsigned;
  unsigned n_elements_of_b_to_load_per_workitem = uninitialised_unsigned;
  unsigned split_on_k = uninitialised_unsigned;
  //TODO : rename to reflect alpha kernel
  unsigned does_beta_c_inc = uninitialised_unsigned;
  unsigned macro_tile_height_and_pad = uninitialised_unsigned;
  unsigned macro_tile_width_and_pad = uninitialised_unsigned;
  unsigned n_elements_in_padded_a_unroll = uninitialised_unsigned;
  unsigned n_elements_in_padded_b_unroll = uninitialised_unsigned;
  unsigned n_micro_tiles_vertically = uninitialised_unsigned;
  unsigned n_micro_tiles_horizontally = uninitialised_unsigned;
  unsigned micro_a_tile_pll_unroll = uninitialised_unsigned; 
  unsigned micro_a_tile_perp_unroll = uninitialised_unsigned; 
  unsigned micro_b_tile_pll_unroll = uninitialised_unsigned;
  unsigned micro_b_tile_perp_unroll = uninitialised_unsigned;    
  unsigned n_micro_a_tiles_pll_unroll = uninitialised_unsigned;  
  unsigned n_micro_b_tiles_pll_unroll = uninitialised_unsigned;
  unsigned preshift_bottommost_tile_height = uninitialised_unsigned;
  unsigned preshift_rightmost_tile_width = uninitialised_unsigned;
  unsigned n_groups_vertically = uninitialised_unsigned;
  unsigned n_groups_horizontally = uninitialised_unsigned;
  //TODO : rename to reflect alpha kernel
  unsigned n_work_groups = uninitialised_unsigned;
  //TODO : rename to reflect alpha kernel
  unsigned global_work_size = uninitialised_unsigned; 

  GA3Params ga3;
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
