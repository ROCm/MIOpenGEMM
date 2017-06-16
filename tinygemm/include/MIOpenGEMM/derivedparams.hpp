#ifndef _DERIVEDPARAMS_HPP
#define _DERIVEDPARAMS_HPP

#include <vector>
#include <string>
#include <map>
#include <limits>
#include <tuple>

#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/geometry.hpp>

namespace MIOpenGEMM{
namespace derivedparams{

const unsigned uninitialised_unsigned = std::numeric_limits<unsigned>::max();

std::tuple<bool, std::string>
get_deriveability(const hyperparams::HyperParams & hp, const Geometry & gg);

class ChiralDerivedParams{
public:

  unsigned macro_tile_length = uninitialised_unsigned;
  unsigned n_elements_in_unroll = uninitialised_unsigned; 
  unsigned main_n_elements_to_load_per_workitem = uninitialised_unsigned;
  unsigned main_n_elements_in_padded_unroll = uninitialised_unsigned;
  unsigned main_micro_tile_pll_unroll = uninitialised_unsigned;
  unsigned main_micro_tile_perp_unroll = uninitialised_unsigned;
  unsigned main_n_micro_tiles_pll_unroll = uninitialised_unsigned;
  unsigned main_macro_tile_length_and_pad = uninitialised_unsigned;
  unsigned main_n_micro_in_macro = uninitialised_unsigned;
  unsigned preshift_final_tile = uninitialised_unsigned;
  
  /* how many macro_lengths to cover m (a) or n (b) */
  unsigned n_groups = uninitialised_unsigned;
  
  /* used when loading LDS -> registers, depends on MIW */
  unsigned main_c_interweave_stride;
  
  /* copy to workspace specific parameters */
  unsigned cw_global_offset = uninitialised_unsigned;
  unsigned cw_n_elements = uninitialised_unsigned;

  /* copy to workspace, type 1, specific parameters */
  unsigned cw1_smallest_possible_ldx = uninitialised_unsigned;
  unsigned cw1_target_ldx = uninitialised_unsigned; 
  size_t cw1_local_work_size = uninitialised_unsigned;
  size_t cw1_work_per_thread = uninitialised_unsigned;
  
  /* copy to workspace, type 2, specific parameters */
  size_t cw2_local_work_size = uninitialised_unsigned;  
  unsigned cw2_load_pll_to_unroll = uninitialised_unsigned; //always perp
  unsigned cw2_micro_tile_pll_unroll = uninitialised_unsigned;
  unsigned cw2_micro_tile_perp_unroll = uninitialised_unsigned;
  unsigned cw2_n_micro_tiles_pll_unroll = uninitialised_unsigned;
  unsigned cw2_n_micro_tiles_perp_unroll = uninitialised_unsigned;
  
  unsigned cw2_n_elements_perp_unroll = uninitialised_unsigned;
  unsigned cw2_n_elements_to_load_per_workitem = uninitialised_unsigned;

        
};

/* all derived parameters */
class DerivedParams{

private:
  const hyperparams::HyperParams & hp;
  const Geometry & gg;

  ChiralDerivedParams adps;
  ChiralDerivedParams bdps;
  void initialise_chis();

  void reset_ga3_params();

  void reset_cw_params(nsHP::eMat emat_x);
  
public:
  
  
  /* initiate all parameters, throwing an error if there is an incompatibility */
  DerivedParams(const hyperparams::HyperParams & hp, const Geometry & gg);
  
  DerivedParams(const hyperparams::HyperParams & hp, const Geometry & gg, std::string s);
  

  std::vector<ChiralDerivedParams * > chis;
  
  ChiralDerivedParams & at(nsHP::eMat emat_x){
    return *chis[emat_x];
  }
  const ChiralDerivedParams & at(nsHP::eMat emat_x) const{
    return *chis[emat_x];
  }
    
  /* does the minimum setting to confirm compatibitily. called by get_deriveability */
  std::tuple<bool, std::string> set_fragile();
  
  unsigned main_macro_tile_area = uninitialised_unsigned;
  unsigned main_micro_tile_area = uninitialised_unsigned;
  
  unsigned main_n_work_items_per_workgroup = uninitialised_unsigned;
  unsigned main_n_work_groups = uninitialised_unsigned;
  unsigned main_global_work_size = uninitialised_unsigned; 
  
  unsigned main_split_on_k = uninitialised_unsigned;
  unsigned main_does_beta_c_inc = uninitialised_unsigned;
  unsigned main_use_edge_trick = uninitialised_unsigned;
  unsigned main_final_fractional_unroll = uninitialised_unsigned;

  /* specific to scaling kernel, betac */
  size_t betac_local_work_size = uninitialised_unsigned;
  size_t betac_work_per_thread = uninitialised_unsigned;


  unsigned cw2_n_macro_tiles_pll_unroll = uninitialised_unsigned;  

  /*the int type for atomics */
  std::string infa;
  /* the function to use for atomic ints */
  std::string fati;
  /* one of __K_NORMAL_FORM__   __K__  and  k_plus_offset */
  std::string effective_k_varies_string; 
  /* pragma unroll string : #pragma unroll\n or "" */
  std::string pragma_unroll_string;
  //* currently one of "float" and "double", set from float_size */
  std::string t_float;
  

  /* GA 3 specific derived parameters */
  unsigned ga3_super_column_width = uninitialised_unsigned;
  unsigned ga3_last_super_column_width = uninitialised_unsigned;


  unsigned get_target_ld(nsHP::eMat emat_x)  const;

  unsigned get_n_elements_in_x_unroll(char x);  
  
  unsigned get_stride(nsHP::eMat emat_x, bool pll_k, bool is_macro, unsigned workspace_type) const;

  unsigned get_stride_cw0(nsHP::eMat emat_x, bool pll_k) const;

  unsigned get_stride_cw1(nsHP::eMat emat_x, bool pll_k) const;

  unsigned get_stride_cw2(nsHP::eMat emat_x, bool pll_k, bool is_macro) const;

  void set_should_be_hyperparams();


};

}
}


#endif
