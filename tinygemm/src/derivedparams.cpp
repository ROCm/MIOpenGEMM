#include <tinygemm/derivedparams.hpp>
#include <tinygemm/mapkeycheck.hpp>
#include <tinygemm/tiling.hpp>


#include <iostream>

namespace tinygemm{
namespace derivedparams{


unsigned DerivedParams::get_target_ld(char x) const{
  return at(x).cw_target_ldx;
}


  
unsigned get_target(unsigned grid_size, unsigned above_distance, unsigned x){
  unsigned to_grid_line = (x - above_distance) / grid_size  + ((x - above_distance) % grid_size != 0);
  return grid_size * to_grid_line + above_distance;
} 


const ChiralDerivedParams & DerivedParams::at(char x) const{
  if (x == 'a'){
    return adps;
  }
  else if (x == 'b'){
    return bdps;
  }
  else{
    throw tinygemm_error("unrecognised char om   ChiralDerivedParams & at(char x)");
  }
}


/* This is the Scott Meyers solution */
ChiralDerivedParams & DerivedParams::at(char x) {
   return const_cast<ChiralDerivedParams &>(static_cast<const DerivedParams &>(*this).at(x));
}



//void DerivedParams::reset_nf_params(){//const tinygemm::TinyGemmGeometry & gg, const hyperparams::HyperParams & hp){
  //nf_k_normal_form = hp.unroll*(gg.k / hp.unroll + ((gg.k % hp.unroll) != 0));
//}


unsigned get_copy_pad(char x){
  if (x == 'a'){
    return 3;
  }
  else{
    return 6;
  }
}
  

void DerivedParams::reset_cw_params(char x){
  
  if (x == 'b' && hp.aps.copy_type == 1 && adps.cw_target_ldx == uninitialised_unsigned){
    throw tinygemm_error("make sure reset_acw1_params is called before reset_bcw1_params, we need that adps.cw_target_ldx be set here in derivedparams reset of bcw1");
  }
  
  at(x).cw_smallest_possible_ldx = gg.coal_is_pll_k(x) ? gg.k : gg.get_non_k_dim(x);
  at(x).cw_target_ldx = get_target(16, get_copy_pad(x), at(x).cw_smallest_possible_ldx);
  at(x).cw_global_offset = (x == 'b' && hp.aps.copy_type == 1) ? get_target_ld('a')*gg.get_uncoal('a') : 0;
  
}
  
void DerivedParams::reset_ga3_params(){
  
  if (main_split_on_k == 1){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.n_target_active_workgroups) / static_cast<double>(hp.n_work_items_per_c_elm))));
  }
  else if (main_split_on_k == 0){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.n_target_active_workgroups))));
  }
  else{
    throw tinygemm_error("main_split_on_k is neither 0 nor 1, how can this be? Logic error in append_super_column_width_defn");
  }  
  ga3_last_super_column_width = bdps.main_n_groups % ga3_super_column_width;
}


std::tuple<bool, std::string>
get_deriveability(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg){
  DerivedParams dp(hp, gg, "uninitialised");
  return dp.set_fragile();
}

DerivedParams::DerivedParams(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, std::string s):hp(hp_), gg(gg_){
  if (s.compare("uninitialised") != 0){
    throw tinygemm_error("the only string with which a DerivedParams object can be initialised is `uninitialised'");
  }
}


std::tuple<bool, std::string>
DerivedParams::set_fragile(){

  main_macro_tile_area = hp.bps.macro_tile_length * hp.aps.macro_tile_length;
  main_micro_tile_area = hp.bps.micro_tile_length * hp.aps.micro_tile_length;
  
  main_n_work_items_per_workgroup = main_macro_tile_area / main_micro_tile_area;

  unsigned required_workspace = 0;

  std::stringstream set_status_ss;
  
  for (char x : {'a', 'b'}){
    at(x).n_elements_in_unroll = hp.at(x).macro_tile_length * hp.unroll;
    at(x).main_n_elements_to_load_per_workitem = at(x).n_elements_in_unroll / main_n_work_items_per_workgroup;  

    if (hp.at(x).copy_type == 1){
      reset_cw_params(x);//gg, hp, x);
      required_workspace += at(x).cw_target_ldx*gg.get_uncoal(x);
    }
  
    /* check 0 : macro tile not too large */  
    if (gg.get_non_k_dim(x) < hp.at(x).macro_tile_length){
      set_status_ss << "gg.get_non_k_dim(x) < hp.at(x).macro_tile_length, this means the tile is too big to work with x = " << x << " . not considering this kernel\n";
    }

  }
  
 
  /* check -1 : enough workspace memory */
  if (gg.workspace_size < required_workspace){
    set_status_ss << "gg.workspace_size ( " << gg.workspace_size << " ) is less then the required workspace ( " << required_workspace << " ) "; 
  }
  
  if (set_status_ss.str() != ""){
    return std::make_tuple(false, set_status_ss.str()); 
  }
  
  /* check 1 : n_work_items_per_workgroup divides n_elements_in_unroll for a and b  */
  for (char x : {'a', 'b'}){
    if (at(x).n_elements_in_unroll % main_n_work_items_per_workgroup != 0){
      set_status_ss << "this is not supported:\nn_work_items_per_workgroup (" << main_n_work_items_per_workgroup << ") is not a factor of n_elements_in_" <<  x << "_UNROLL (" << adps.n_elements_in_unroll << "). \nConsider rounding unroll up. \n";
      return std::make_tuple(false, set_status_ss.str());
    }
  }
  
  
  /* check 2 : tilability */
  for (char x : {'a', 'b'}){  
    auto tup = tiling::get_tileability(hp.at(x).macro_tile_length, hp.unroll, at(x).main_n_elements_to_load_per_workitem);
    if (std::get<0>(tup) == false){
      return tup;
    }
  }
  
  /* ran the gauntlet, returning deriveable is true */
  return std::make_tuple(true, "");
}


DerivedParams::DerivedParams(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_): hp(hp_), gg(gg_) {

  auto tup = set_fragile();

  if (std::get<0>(tup) == false){
    throw tinygemm_error("Failure to construct DerivedParams. Problem caught in set_fragile. It is recommended to run use function derivable to check that a valid DerivedParams can be constructed. The message returned in set_fragile is :  " + std::get<1>(tup));
  }
  
  /* do the tiling */  
  tiling::set_tile_dimensions(adps.main_micro_tile_perp_unroll, adps.main_micro_tile_pll_unroll, hp.aps.macro_tile_length, hp.unroll, adps.main_n_elements_to_load_per_workitem, hp.aps.load_pll_to_unroll == 0); 
  
  tiling::set_tile_dimensions(bdps.main_micro_tile_perp_unroll, bdps.main_micro_tile_pll_unroll, hp.bps.macro_tile_length, hp.unroll, bdps.main_n_elements_to_load_per_workitem, hp.bps.load_pll_to_unroll == 0); 
    
  main_split_on_k = hp.n_work_items_per_c_elm == 1 ? 0 : 1;
  main_does_beta_c_inc = main_split_on_k == 1 ? 0 : 1; 
  
  adps.main_c_interweave_stride = hp.c_micro_tiles_interwoven == 0 ? 1 : adps.main_n_micro_in_macro;
  bdps.main_c_interweave_stride = hp.c_micro_tiles_interwoven == 0 ? 1 : bdps.main_n_micro_in_macro;
  
  if (hp.n_work_items_per_c_elm == 1){
    infa = "n_work_items_per_c_elm is 1, should not be using atomics";
    fati = "n_work_items_per_c_elm is 1, should not be using atomics";
  }
  
  else{
    infa = gg.derived.float_size_bits == 32 ? "uint" : "ulong";
    fati = gg.derived.float_size_bits == 32 ? "atomic_cmpxchg" : "atom_cmpxchg";
  }
  
  pragma_unroll_string = hp.unroll_pragma == 1 ?  "#pragma unroll\n" : "" ;
  
  if (hp.normal_form != 0 && hp.normal_form != 1){
    throw tinygemm_error("hp.normal_form is not set yet, reorder needed in derivedparams");
  }
  
  effective_k_varies_string = hp.unroll_for_offset == 0 ? "__K__" : "k_plus_offset";
  t_float = gg.derived.float_size_bits == 32 ? "float" : "double";
  
  
  for (char x : {'a', 'b'}){
    at(x).main_preshift_final_tile = 1 + (gg.get_non_k_dim(x) - 1) % hp.at(x).macro_tile_length;
    at(x).main_n_groups = gg.get_non_k_dim(x) / hp.at(x).macro_tile_length + (at(x).main_preshift_final_tile != hp.at(x).macro_tile_length);
    at(x).main_macro_tile_length_and_pad = hp.at(x).macro_tile_length + hp.at(x).lds_pad_size;
    at(x).main_n_elements_in_padded_unroll = at(x).main_macro_tile_length_and_pad * hp.unroll;
    at(x).main_n_micro_in_macro = hp.at(x).macro_tile_length / hp.at(x).micro_tile_length;
    at(x).main_n_micro_tiles_pll_unroll = hp.unroll / at(x).main_micro_tile_pll_unroll;
  
  }
  
  main_n_work_groups = hp.n_work_items_per_c_elm * 
  ((gg.m/hp.aps.macro_tile_length) + (gg.m%hp.aps.macro_tile_length != 0)) * 
  ((gg.n/hp.bps.macro_tile_length) + (gg.n%hp.bps.macro_tile_length != 0));
  
  main_global_work_size = main_n_work_groups * main_n_work_items_per_workgroup;
  


    
  if (hp.group_allocation == 3){
    reset_ga3_params();
  }
  
  
  
  /* these guys are hyper params, with a check if not optional ? */
  main_use_edge_trick = 1;
  main_final_fractional_unroll = (hp.unroll_for_offset == 1 || gg.k%hp.unroll != 0) ? 1 : 0;


}


unsigned DerivedParams::get_n_elements_in_x_unroll(char x){
  if (x == 'a'){
    return adps.n_elements_in_unroll;
  }
  else if (x == 'b'){
    return bdps.n_elements_in_unroll;
  }
  else{
    throw tinygemm_error("unrecognised x in get_n_elements_in_x_unroll");
  }
}


unsigned DerivedParams::get_stride(char x, bool pll_k, bool is_macro) const{


  //if (hp.normal_form == 0){
    //adps.main_effective_ldx = hp.aps.copy_type == 0 ? gg.lda : adps.cw_target_ldx;
    //bdps.main_effective_ldx = hp.bps.copy_type == 0 ? gg.ldb : bdps.cw_target_ldx;
  //}
 

  if (hp.normal_form == 0){
    unsigned effective_ldx = hp.at(x).copy_type == 0  ? gg.get_ld(x) : at(x).cw_target_ldx;
    return gg.coal_is_pll_k(x) == pll_k ? 1 : effective_ldx; //at(x).main_effective_ldx;
  }
  
  else{
    if (is_macro == false){
      return pll_k == true ? hp.at(x).macro_tile_length : 1;
    }
    else{
      return pll_k == true ? hp.at(x).macro_tile_length : gg.k;
    }
  }
}



}
}
