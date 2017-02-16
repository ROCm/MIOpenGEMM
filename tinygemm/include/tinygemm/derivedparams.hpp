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


/* all derived parameters */
class DerivedParams{

private:
  void reset_ga3_params(const hyperparams::HyperParams & hp);
  void reset_acw1_params(const tinygemm::TinyGemmGeometry & gg);
  void reset_bcw1_params(const tinygemm::TinyGemmGeometry & gg, const hyperparams::HyperParams & hp);
  void reset_nf_params(const tinygemm::TinyGemmGeometry & gg, const hyperparams::HyperParams & hp);
  
public:
  
  /* initiate all parameters, throwing an error if there is an incompatibility */
  DerivedParams(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  
  DerivedParams(std::string s);
  
    
  /* does the minimum setting to confirm compatibitily. called by get_deriveability */
  std::tuple<bool, std::string> set_fragile(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  
  /* TODO : write descriptions */
  unsigned macro_tile_area = uninitialised_unsigned;
  unsigned micro_tile_area = uninitialised_unsigned;
  unsigned main_n_work_items_per_workgroup = uninitialised_unsigned;
  unsigned n_elements_in_a_unroll = uninitialised_unsigned;
  unsigned n_elements_in_b_unroll = uninitialised_unsigned;
  unsigned n_elements_of_a_to_load_per_workitem = uninitialised_unsigned;
  unsigned n_elements_of_b_to_load_per_workitem = uninitialised_unsigned;
  unsigned split_on_k = uninitialised_unsigned;
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
  unsigned main_n_work_groups = uninitialised_unsigned;
  unsigned main_global_work_size = uninitialised_unsigned; 
 
  /* used when loading LDA -> registers, depends on MIW*/
  std::string strided_i_vertical;
  /* used when loading LDA -> registers, depends on MIW*/
  std::string strided_i_horizontal;
  /*the int type for atomics */
  std::string infa;
  /* the function to use for atomic ints */
  std::string fati;
  /* one of __K_NORMAL_FORM__   __K__  and  k_plus_offset */
  std::string effective_k_varies_string; 
  /* indexinf of c, depends on MIW*/
  std::string alpha_scaled;
  /* pragma unroll string : #pragma unroll\n or "" */
  std::string pragma_unroll_string;
  /* currently one of "float" and "double", set from float_size */
  std::string t_float;

  /* GA 3 specific derived parameters */
  unsigned ga3_super_column_width = uninitialised_unsigned;
  unsigned ga3_last_super_column_width = uninitialised_unsigned;
  
  /* ACW1 specific derived parameters */
  unsigned acw1_smallest_possible_lda = uninitialised_unsigned;
  unsigned acw1_target_lda = uninitialised_unsigned;
  
  /* BCW1 specific derived parameters */
  unsigned bcw1_smallest_possible_ldb = uninitialised_unsigned;
  unsigned bcw1_target_ldb = uninitialised_unsigned; 
  unsigned bcw1_global_offset_b = uninitialised_unsigned;

  unsigned get_target_ld(char c) const;


  /* normal form specifics */
  unsigned nf_k_normal_form = uninitialised_unsigned;
  unsigned needs_final_fractional_unroll = uninitialised_unsigned;


  /* either original ldas, or those targetted in copy workspace */
  unsigned effective_lda = uninitialised_unsigned;
  unsigned effective_ldb = uninitialised_unsigned;



};

}
}


#endif
