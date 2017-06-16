#include <MIOpenGEMM/derivedparams.hpp>
#include <MIOpenGEMM/error.hpp>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <MIOpenGEMM/tiling.hpp>


#include <iostream>

namespace MIOpenGEMM{
namespace derivedparams{

unsigned DerivedParams::get_target_ld(nsHP::eMat emat_x) const{
  return at(emat_x).cw1_target_ldx;
}


  
unsigned get_target(unsigned grid_size, unsigned above_distance, unsigned x){
  unsigned to_grid_line = (x - above_distance) / grid_size  + ((x - above_distance) % grid_size != 0);
  return grid_size * to_grid_line + above_distance;
} 


unsigned get_copy_pad(nsHP::eMat emat_x){
  if (emat_x == nsHP::matA){
    return 3;
  }
  else{
    return 6;
  }
}
 

void DerivedParams::reset_cw_params(nsHP::eMat emat_x){
  if (emat_x == nsHP::matB && hp.at(nsHP::matA).vs[nsHP::WOS] != 0 && adps.cw_n_elements == uninitialised_unsigned){
    throw miog_error(std::string("make sure reset acw1 params is called before reset_bcw1_params, we need that adps.cw1_target_ldx be set here in derivedparams reset of bcw1"));
  }
  
  /* simple copy with padding */
  if (hp.at( emat_x ).vs[nsHP::WOS] == 1){
    at(emat_x).cw1_smallest_possible_ldx = gg.coal_is_pll_k(emat_x) ? gg.k : gg.get_non_k_dim(emat_x);
    at(emat_x).cw1_target_ldx = get_target(16, get_copy_pad(emat_x), at(emat_x).cw1_smallest_possible_ldx);
    at(emat_x).cw_n_elements = at(emat_x).cw1_target_ldx*gg.get_uncoal(emat_x);
  }
  
  else if (hp.at( emat_x ).vs[nsHP::WOS] == 2){
    
    at(emat_x).cw2_n_elements_perp_unroll = at(emat_x).n_groups*at(emat_x).macro_tile_length;    
    at(emat_x).cw_n_elements = at(emat_x).cw2_n_elements_perp_unroll * gg.k;
    
    cw2_n_macro_tiles_pll_unroll = gg.k / hp.at(nsHP::matC).vs[nsHP::UNR] + ((gg.k % hp.at(nsHP::matC).vs[nsHP::UNR]) != 0);
    
  }

  else{
    throw miog_error("copy type is neither 1 nor 2, so can't be correct that there's a call to reset_cw_params");
  }
  
  at(emat_x).cw_global_offset = (emat_x == nsHP::matB && hp.at(nsHP::matA).vs[nsHP::WOS] != 0) ? at(nsHP::matA).cw_n_elements : 0;
  
}
  
void DerivedParams::reset_ga3_params(){

  if (main_split_on_k == 1){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.at(nsHP::matC).vs[nsHP::NAW]) / static_cast<double>(hp.at(nsHP::matC).vs[nsHP::ICE]))));
  }
  else if (main_split_on_k == 0){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.at(nsHP::matC).vs[nsHP::NAW]))));
  }
  else{
    throw miog_error("main_split_on_k is neither 0 nor 1, how can this be? Logic error in reset_ga3_params");
  }  
  ga3_last_super_column_width = bdps.n_groups % ga3_super_column_width;
}


std::tuple<bool, std::string>
get_deriveability(const hyperparams::HyperParams & hp, const Geometry & gg){
  DerivedParams dp(hp, gg, "uninitialised");
  return dp.set_fragile();
}


DerivedParams::DerivedParams(const hyperparams::HyperParams & hp_, const Geometry & gg_, std::string s):hp(hp_), gg(gg_){
  
  initialise_chis();
  
  if (s.compare("uninitialised") != 0){
    throw miog_error("the only string with which a DerivedParams object can be initialised is `uninitialised'");
  }
}


std::tuple<bool, std::string>
DerivedParams::set_fragile(){
  
  set_should_be_hyperparams();

  auto grid_size_tuple = nsMAC::get_mac_grid(hp.at(nsHP::matC).vs[nsHP::MAC], hp.at(nsHP::matC).vs[nsHP::SKW]);
  if (std::get<0> (grid_size_tuple) == false){
    return std::make_tuple(false, std::get<1> (grid_size_tuple)); 
  }
  
  auto grid_size = std::get<2>(grid_size_tuple);
  
   
  at(nsHP::matA).macro_tile_length = grid_size[nsHP::matA] * hp.at(nsHP::matA).vs[nsHP::MIC];
  at(nsHP::matB).macro_tile_length = grid_size[nsHP::matB] * hp.at(nsHP::matB).vs[nsHP::MIC];
  
  
      
      
  for (auto emat_x : {nsHP::matA, nsHP::matB}){
    at(emat_x).preshift_final_tile = 1 + (gg.get_non_k_dim(emat_x) - 1) % at(emat_x).macro_tile_length;
    at(emat_x).n_groups = gg.get_non_k_dim(emat_x) / at(emat_x).macro_tile_length + (at(emat_x).preshift_final_tile != at(emat_x).macro_tile_length);
    at(emat_x).main_macro_tile_length_and_pad = at(emat_x).macro_tile_length + hp.at( emat_x ).vs[nsHP::PAD];
    at(emat_x).main_n_elements_in_padded_unroll = at(emat_x).main_macro_tile_length_and_pad * hp.at(nsHP::matC).vs[nsHP::UNR];
  }
  

  main_macro_tile_area = at(nsHP::matB).macro_tile_length * at(nsHP::matA).macro_tile_length;
  main_micro_tile_area = hp.at(nsHP::matB).vs[nsHP::MIC] * hp.at(nsHP::matA).vs[nsHP::MIC];
  main_n_work_items_per_workgroup = main_macro_tile_area / main_micro_tile_area;

  unsigned required_workspace = 0;

  std::stringstream set_status_ss;

  for (auto emat_x : {nsHP::matA, nsHP::matB}){
    /* check - 3 : the macro tile is too tall */
    if (gg.m < at(nsHP::matA).macro_tile_length){
      set_status_ss  << "m < aps.macro_tile_length, not considering this kernel. ";
    }
    
    /* check - 4 : the macro tile is too wide */
    else if (gg.n < at(nsHP::matB).macro_tile_length){
      set_status_ss << "m < bps.macro_tile_length, not considering this kernel. ";
    }
    
    at(emat_x).n_elements_in_unroll = at(emat_x).macro_tile_length * hp.at(nsHP::matC).vs[nsHP::UNR];
    at(emat_x).main_n_elements_to_load_per_workitem = at(emat_x).n_elements_in_unroll / main_n_work_items_per_workgroup;  
    
    if (hp.at( emat_x ).vs[nsHP::WOS] == 2){
      at(emat_x).cw2_n_elements_to_load_per_workitem = at(emat_x).n_elements_in_unroll / at(emat_x).cw2_local_work_size;  
    }
    
    if (hp.at( emat_x ).vs[nsHP::WOS] != 0){
      reset_cw_params(emat_x);
      required_workspace += at(emat_x).cw_n_elements; //cw1_target_ldx*gg.get_uncoal(x);
    }
  
    /* check 0 : macro tile not too large */  
    if (gg.get_non_k_dim(emat_x) < at(emat_x).macro_tile_length){
      set_status_ss << "gg.get_non_k_dim( " << matChars[emat_x] << " )  < at ( " << matChars[emat_x] << " ).macro_tile_length, this means the tile is too big to work with  " << matChars[emat_x] << " . not considering this kernel. ";
    }

  }
  

 
  /* check -1 : enough workspace memory */
  if (gg.workspace_size < required_workspace){
    set_status_ss << "gg.workspace_size ( " << gg.workspace_size << " ) is less then the required workspace ( " << required_workspace << " ). "; 
  }
  
  if (set_status_ss.str() != ""){
    return std::make_tuple(false, set_status_ss.str()); 
  }
  
  /* check 1 : n_work_items_per_workgroup divides n_elements_in_unroll for a and b  */

  auto is_div = [&set_status_ss, this](nsHP::eMat emat_x, std::string which, unsigned val){

    if (at(emat_x).n_elements_in_unroll % val != 0){
      set_status_ss << "this is not supported: " << which << " (" << val << ") is not a factor of n_elements_in_(" <<  matChars[emat_x] << ")_unroll (" << at(emat_x).n_elements_in_unroll << "). \n" << "Consider rounding unroll up. ";
      return std::make_tuple<bool, std::string>(false, set_status_ss.str());
    }
    else{
      return std::make_tuple<bool, std::string>(true, {});
    }
  };



  for (auto emat_x : {nsHP::matA, nsHP::matB}){

    auto tup = is_div(emat_x, "main_n_work_items_per_workgroup", main_n_work_items_per_workgroup);
    if (std::get<0>(tup) == false){
      return tup;
    }    
    
    if (hp.at( emat_x ).vs[nsHP::WOS] == 2){// && at(emat_x).n_elements_in_unroll % at(emat_x).cw2_local_work_size != 0){
      auto tup_cw = is_div(emat_x, "at(emat_x).cw2_local_work_size", at(emat_x).cw2_local_work_size);
      if (std::get<0>(tup_cw) == false){
        return tup_cw;
      }
    }
  }
  
  
  /* check 2 : tileability */
  for (auto emat_x : {nsHP::matA, nsHP::matB}){

    auto tup = tiling::get_tileability(at(emat_x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(emat_x).main_n_elements_to_load_per_workitem);
    if (std::get<0>(tup) == false){
      return tup;
    }
    
    if (hp.at( emat_x ).vs[nsHP::WOS] == 2){
      tup = tiling::get_tileability(at(emat_x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(emat_x).cw2_n_elements_to_load_per_workitem);
      if (std::get<0>(tup) == false){
        return tup;
      }
    }
  }
  
  if (hp.at(nsHP::matC).vs[nsHP::UFO] == nsHP::yes){
    if (gg.k <= hp.at(nsHP::matC).vs[nsHP::UNR]){
      return std::make_tuple(false, "UFO = yes, so UNR must be greater that k");
    }
  }
  
  /* ran the gauntlet, returning deriveable is true */
  return std::make_tuple(true, "");
}


void DerivedParams::initialise_chis(){
  chis.resize(2);
  if (nsHP::matA > 2 || nsHP::matB > 2){
    throw miog_error("In DeriverParams constructor, enums too large (strange)");
  }
  
  chis[nsHP::matA] = &adps;
  chis[nsHP::matB] = &bdps;
}
 
 
DerivedParams::DerivedParams(const hyperparams::HyperParams & hp_, const Geometry & gg_): hp(hp_), gg(gg_) {

  initialise_chis();
  
  auto tup = set_fragile();

  if (std::get<0>(tup) == false){
    throw miog_error("Failure to construct DerivedParams. Problem caught in set_fragile. It is recommended to run function ` derivable ' to check that a valid DerivedParams can be constructed. The message returned in set_fragile is :  " + std::get<1>(tup));
  }
  
  /* do the tiling */  

  for (auto emat_x : {nsHP::matA, nsHP::matB}){
      
    tiling::set_tile_dimensions(at(emat_x).main_micro_tile_perp_unroll, at(emat_x).main_micro_tile_pll_unroll, at(emat_x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(emat_x).main_n_elements_to_load_per_workitem, hp.at( emat_x ).vs[nsHP::PLU] == 0);
    
    if (hp.at( emat_x ).vs[nsHP::WOS] == 2){
      tiling::set_tile_dimensions(at(emat_x).cw2_micro_tile_perp_unroll, at(emat_x).cw2_micro_tile_pll_unroll, at(emat_x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(emat_x).cw2_n_elements_to_load_per_workitem, at(emat_x).cw2_load_pll_to_unroll == 0);
    }
  } 
  
  
  main_split_on_k = hp.at(nsHP::matC).vs[nsHP::ICE] == 1 ? 0 : 1;
  main_does_beta_c_inc = main_split_on_k == 1 ? 0 : 1; 
  
  if (hp.at(nsHP::matC).vs[nsHP::ICE] == 1){
    infa = "n_work_items_per_c_elm is 1, should not be using atomics";
    fati = "n_work_items_per_c_elm is 1, should not be using atomics";
  }
  
  else{
    infa = gg.derived.float_size_bits == 32 ? "uint" : "ulong";
    fati = gg.derived.float_size_bits == 32 ? "atomic_cmpxchg" : "atom_cmpxchg";
  }
  
  pragma_unroll_string = hp.at(nsHP::matC).vs[nsHP::PUN] == 1 ?  "#pragma unroll\n" : "" ;
  
  effective_k_varies_string = hp.at(nsHP::matC).vs[nsHP::UFO] == 0 ? "__K__" : "k_plus_offset";
  t_float = gg.derived.float_size_bits == 32 ? "float" : "double";
  
  main_n_work_groups = hp.at(nsHP::matC).vs[nsHP::ICE] * 
  ((gg.m/at(nsHP::matA).macro_tile_length) + (gg.m%at(nsHP::matA).macro_tile_length != 0)) * 
  ((gg.n/at(nsHP::matB).macro_tile_length) + (gg.n%at(nsHP::matB).macro_tile_length != 0));
  
  main_global_work_size = main_n_work_groups * main_n_work_items_per_workgroup;
  
  for (auto emat_x : {nsHP::matA, nsHP::matB}){

    at(emat_x).main_n_micro_in_macro = at(emat_x).macro_tile_length / hp.at( emat_x ).vs[nsHP::MIC];
    at(emat_x).main_n_micro_tiles_pll_unroll = hp.at(nsHP::matC).vs[nsHP::UNR] / at(emat_x).main_micro_tile_pll_unroll;
    at(emat_x).main_c_interweave_stride = hp.at( emat_x ).vs[nsHP::MIW] == 0 ? 1 : at(emat_x).main_n_micro_in_macro;  
    
    if (hp.at( emat_x ).vs[nsHP::WOS] == 2){
      at(emat_x).cw2_n_micro_tiles_pll_unroll = hp.at(nsHP::matC).vs[nsHP::UNR] / at(emat_x).cw2_micro_tile_pll_unroll;
      at(emat_x).cw2_n_micro_tiles_perp_unroll = at(emat_x).macro_tile_length / at(emat_x).cw2_micro_tile_perp_unroll;
    }
  }

  if (hp.at(nsHP::matC).vs[nsHP::GAL] == 3){
    reset_ga3_params();
  }

  /* these guys are hyper params, with a check if not optional ? */
  main_use_edge_trick = (gg.m%at(nsHP::matA).macro_tile_length == 0 && gg.n%at(nsHP::matB).macro_tile_length == 0) ? 0 : 1;
  main_final_fractional_unroll = (hp.at(nsHP::matC).vs[nsHP::UFO] == 1 || gg.k%hp.at(nsHP::matC).vs[nsHP::UNR] != 0) ? 1 : 0;
  
}

/* TODO : move to hyper params */
void DerivedParams::set_should_be_hyperparams(){
  
  betac_local_work_size = 256;
  betac_work_per_thread = 2;

  for (auto emat_x : {nsHP::matA, nsHP::matB}){

    at(emat_x).cw1_local_work_size = 256;
    at(emat_x).cw1_work_per_thread = 2;
    
    at(emat_x).cw2_load_pll_to_unroll = 0;    
    at(emat_x).cw2_local_work_size = 64;
  }
}

unsigned DerivedParams::get_n_elements_in_x_unroll(char x){
  if (x == 'a'){
    return adps.n_elements_in_unroll;
  }
  else if (x == 'b'){
    return bdps.n_elements_in_unroll;
  }
  else{
    throw miog_error("unrecognised x in get_n_elements_in_x_unroll");
  }
}


unsigned DerivedParams::get_stride(nsHP::eMat emat_x, bool pll_k, bool is_macro, unsigned workspace_type_) const{
  
  if (workspace_type_ == 0){
    return get_stride_cw0(emat_x, pll_k);
  }
  
  else if (workspace_type_ == 1){
    return get_stride_cw1(emat_x, pll_k);
  }
  
  else if (workspace_type_ == 2){
    return get_stride_cw2(emat_x, pll_k, is_macro);
  }
  else throw miog_error("unrecognised workspace_type in get_strinde in derivedparams");
}

unsigned DerivedParams::get_stride_cw0(nsHP::eMat emat_x, bool pll_k) const{
  return gg.coal_is_pll_k(emat_x) == pll_k ? 1 : gg.ldX.at(emat_x); 
}

unsigned DerivedParams::get_stride_cw1(nsHP::eMat emat_x, bool pll_k) const{
  return gg.coal_is_pll_k(emat_x) == pll_k ? 1 : at(emat_x).cw1_target_ldx; 
}

unsigned DerivedParams::get_stride_cw2(nsHP::eMat emat_x, bool pll_k, bool is_macro) const{
  if (is_macro == false){
    return pll_k == true ? at(emat_x).macro_tile_length : 1;
  }
  else{
    return pll_k == true ? at(emat_x).macro_tile_length : gg.k;
  }
  
}

}
}
