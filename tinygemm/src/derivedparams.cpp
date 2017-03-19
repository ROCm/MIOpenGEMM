#include <tinygemm/derivedparams.hpp>
#include <tinygemm/mapkeycheck.hpp>
#include <tinygemm/tiling.hpp>


#include <iostream>

namespace tinygemm{
namespace derivedparams{


unsigned DerivedParams::get_target_ld(char x) const{
  if (x != 'a' && x != 'A' && x != 'b' && x != 'B'){
    throw tinygemm_error("call to get_target_ld must be made with an aAbB, not " + std::string(1, x));  
  }
  return at(x).cw1_target_ldx;
}


  
unsigned get_target(unsigned grid_size, unsigned above_distance, unsigned x){
  unsigned to_grid_line = (x - above_distance) / grid_size  + ((x - above_distance) % grid_size != 0);
  return grid_size * to_grid_line + above_distance;
} 


const ChiralDerivedParams & DerivedParams::at(char x) const{
  if (x == 'a' || x == 'A'){
    return adps;
  }
  else if (x == 'b' || x == 'B'){
    return bdps;
  }
  else{
    throw tinygemm_error("unrecognised char in ChiralDerivedParams & at(char x) : " + x);
  }
}


/* This is the Scott Meyers solution */
ChiralDerivedParams & DerivedParams::at(char x) {
   return const_cast<ChiralDerivedParams &>(static_cast<const DerivedParams &>(*this).at(x));
}




unsigned get_copy_pad(char x){
  if (x == 'a'){
    return 3;
  }
  else{
    return 6;
  }
}
  

void DerivedParams::reset_cw_params(char x){
 
  
  if (x == 'b' && hp.at(nsHP::matA).vs[nsHP::WOS] != 0 && adps.cw_n_elements == uninitialised_unsigned){
    throw tinygemm_error(std::string("make sure reset acw1 params is called before reset_bcw1_params, we need that adps.cw1_target_ldx be set here in derivedparams reset of bcw1"));
  }
  
  
  /* simple copy with padding */
  if (hp.at(x).vs[nsHP::WOS] == 1){
    at(x).cw1_smallest_possible_ldx = gg.coal_is_pll_k(x) ? gg.k : gg.get_non_k_dim(x);
    at(x).cw1_target_ldx = get_target(16, get_copy_pad(x), at(x).cw1_smallest_possible_ldx);
    at(x).cw_n_elements = at(x).cw1_target_ldx*gg.get_uncoal(x);
  }
  
  else if (hp.at(x).vs[nsHP::WOS] == 2){
    
    at(x).cw2_n_elements_perp_unroll = at(x).n_groups*at(x).macro_tile_length;    
    at(x).cw_n_elements = at(x).cw2_n_elements_perp_unroll * gg.k;
    
    cw2_n_macro_tiles_pll_unroll = gg.k / hp.at(nsHP::matC).vs[nsHP::UNR] + ((gg.k % hp.at(nsHP::matC).vs[nsHP::UNR]) != 0);
    
    //std::cout << "\n\n" << at(x).cw_n_elements;
    //std::cout << "\n\n\n\n\nwhat cw2 resets here ? \n\n\n\n" << std::endl;
    ////throw tinygemm_error("currently copy type 2 is not ready mmmmmmm");
  }

  else{
    throw tinygemm_error("copy type is neither 1 not 2, so can't be correct that there's a call to reset_cw_params");
  }
  
  at(x).cw_global_offset = (x == 'b' && hp.at('a').vs[nsHP::WOS] != 0) ? at('a').cw_n_elements : 0;
  
  
  
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
    throw tinygemm_error("main_split_on_k is neither 0 nor 1, how can this be? Logic error in reset_ga3_params");
  }  
  ga3_last_super_column_width = bdps.n_groups % ga3_super_column_width;
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
  
  set_should_be_hyperparams();

  
  
  //TODO : tidy this up, compactify
  if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a4b8)  {
    at('A').macro_tile_length = 4;
    at('B').macro_tile_length = 8;
  }

  else if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a8b4)  {
    at('A').macro_tile_length = 8;
    at('B').macro_tile_length = 4;
  }

  
  else if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a8b8)  {
    at('A').macro_tile_length = 8;
    at('B').macro_tile_length = 8;
  }
  
  else if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a8b16)  {
    at('A').macro_tile_length = 8;
    at('B').macro_tile_length = 16;
  }

  else if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a16b8)  {
    at('A').macro_tile_length = 16;
    at('B').macro_tile_length = 8;
  }
   
  else if (hp.at(nsHP::matC).vs[nsHP::MAC] == nsMAC::a16b16)  {
    at('A').macro_tile_length = 16;
    at('B').macro_tile_length = 16;
  }
  
  else{
    throw tinygemm_error("unrecognised MAC (macro tile sizes cannot be set)");
  }
  
  at('A').macro_tile_length *= hp.at('A').vs[nsHP::MIC];
  at('B').macro_tile_length *= hp.at('B').vs[nsHP::MIC];



  
  
  for (char x : {'a', 'b'}){
    at(x).preshift_final_tile = 1 + (gg.get_non_k_dim(x) - 1) % at(x).macro_tile_length;
    at(x).n_groups = gg.get_non_k_dim(x) / at(x).macro_tile_length + (at(x).preshift_final_tile != at(x).macro_tile_length);
    at(x).main_macro_tile_length_and_pad = at(x).macro_tile_length + hp.at(x).vs[nsHP::MIC];
    at(x).main_n_elements_in_padded_unroll = at(x).main_macro_tile_length_and_pad * hp.at(nsHP::matC).vs[nsHP::UNR];
  }
  

  main_macro_tile_area = at('b').macro_tile_length * at('a').macro_tile_length;
  main_micro_tile_area = hp.at(nsHP::matB).vs[nsHP::MIC] * hp.at(nsHP::matA).vs[nsHP::MIC];
  main_n_work_items_per_workgroup = main_macro_tile_area / main_micro_tile_area;

  unsigned required_workspace = 0;

  std::stringstream set_status_ss;
  
  for (char x : {'a', 'b'}){


    /* check - 3 : the macro tile is too tall */
    if (gg.m < at('a').macro_tile_length){
      set_status_ss  << "m < aps.macro_tile_length, not considering this kernel\n";
    }
    
    /* check - 4 : the macro tile is too wide */
    else if (gg.n < at('b').macro_tile_length){
      set_status_ss << "m < bps.macro_tile_length, not considering this kernel\n";
    }
    
  
    
    at(x).n_elements_in_unroll = at(x).macro_tile_length * hp.at(nsHP::matC).vs[nsHP::UNR];
    at(x).main_n_elements_to_load_per_workitem = at(x).n_elements_in_unroll / main_n_work_items_per_workgroup;  
    
    if (hp.at(x).vs[nsHP::WOS] == 2){
      at(x).cw2_n_elements_to_load_per_workitem = at(x).n_elements_in_unroll / at(x).cw2_local_work_size;  
    }
    
    if (hp.at(x).vs[nsHP::WOS] != 0){
      reset_cw_params(x);
      required_workspace += at(x).cw_n_elements; //cw1_target_ldx*gg.get_uncoal(x);
    }
  
    /* check 0 : macro tile not too large */  
    if (gg.get_non_k_dim(x) < at(x).macro_tile_length){
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

  auto is_div = [&set_status_ss, this](char x, std::string which, unsigned val){
    if (at(x).n_elements_in_unroll % val != 0){
      set_status_ss << "this is not supported:\n" << which << " (" << val << ") is not a factor of n_elements_in_" <<  x << "_unroll (" << at(x).n_elements_in_unroll << "). \n" << "Consider rounding unroll up. \n";
      return std::make_tuple<bool, std::string>(false, set_status_ss.str());
    }
    else{
      return std::make_tuple<bool, std::string>(true, {});
    }
  };


  for (char x : {'a', 'b'}){
    auto tup = is_div(x, "main_n_work_items_per_workgroup", main_n_work_items_per_workgroup);
    if (std::get<0>(tup) == false){
      return tup;
    }    
    
    if (hp.at(x).vs[nsHP::WOS] == 2){// && at(x).n_elements_in_unroll % at(x).cw2_local_work_size != 0){
      auto tup_cw = is_div(x, "at(x).cw2_local_work_size", at(x).cw2_local_work_size);
      if (std::get<0>(tup_cw) == false){
        return tup_cw;
      }
    }
  }
  
  
  /* check 2 : tileability */
  for (char x : {'a', 'b'}){  
    auto tup = tiling::get_tileability(at(x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(x).main_n_elements_to_load_per_workitem);
    if (std::get<0>(tup) == false){
      return tup;
    }
    
    if (hp.at(x).vs[nsHP::WOS] == 2){
      tup = tiling::get_tileability(at(x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(x).cw2_n_elements_to_load_per_workitem);
      if (std::get<0>(tup) == false){
        return tup;
      }
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
  for (char x : {'a', 'b'}){
  
    tiling::set_tile_dimensions(at(x).main_micro_tile_perp_unroll, at(x).main_micro_tile_pll_unroll, at(x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(x).main_n_elements_to_load_per_workitem, hp.at(x).vs[nsHP::PLU] == 0);
    
    if (hp.at(x).vs[nsHP::WOS] == 2){
      tiling::set_tile_dimensions(at(x).cw2_micro_tile_perp_unroll, at(x).cw2_micro_tile_pll_unroll, at(x).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], at(x).cw2_n_elements_to_load_per_workitem, at(x).cw2_load_pll_to_unroll == 0);
    }
  } 
  
  //tiling::set_tile_dimensions(bdps.main_micro_tile_perp_unroll, bdps.main_micro_tile_pll_unroll, hp.at(nsHP::matB).macro_tile_length, hp.at(nsHP::matC).vs[nsHP::UNR], bdps.main_n_elements_to_load_per_workitem, hp.at(nsHP::matB).load_pll_to_unroll == 0); 
  
  
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
  ((gg.m/at('a').macro_tile_length) + (gg.m%at('a').macro_tile_length != 0)) * 
  ((gg.n/at('b').macro_tile_length) + (gg.n%at('b').macro_tile_length != 0));
  
  main_global_work_size = main_n_work_groups * main_n_work_items_per_workgroup;
  
  for (char x : {'a', 'b'}){      
    at(x).main_n_micro_in_macro = at(x).macro_tile_length / hp.at(x).vs[nsHP::MIC];
    at(x).main_n_micro_tiles_pll_unroll = hp.at(nsHP::matC).vs[nsHP::UNR] / at(x).main_micro_tile_pll_unroll;
    at(x).main_c_interweave_stride = hp.at(x).vs[nsHP::MIW] == 0 ? 1 : at(x).main_n_micro_in_macro;  
    
    
    if (hp.at(x).vs[nsHP::WOS] == 2){
      at(x).cw2_n_micro_tiles_pll_unroll = hp.at(nsHP::matC).vs[nsHP::UNR] / at(x).cw2_micro_tile_pll_unroll;
      at(x).cw2_n_micro_tiles_perp_unroll = at(x).macro_tile_length / at(x).cw2_micro_tile_perp_unroll;
    }

  }

  if (hp.at(nsHP::matC).vs[nsHP::GAL] == 3){
    reset_ga3_params();
  }

  /* these guys are hyper params, with a check if not optional ? */
  main_use_edge_trick = (gg.m%at('a').macro_tile_length == 0 && gg.n%at('b').macro_tile_length == 0) ? 0 : 1;
  main_final_fractional_unroll = (hp.at(nsHP::matC).vs[nsHP::UFO] == 1 || gg.k%hp.at(nsHP::matC).vs[nsHP::UNR] != 0) ? 1 : 0;
  
}

void DerivedParams::set_should_be_hyperparams(){
  

  betac_local_work_size = 256;
  betac_work_per_thread = 2;

  for (char x : {'a', 'b'}){

    at(x).cw1_local_work_size = 256;
    at(x).cw1_work_per_thread = 2;
    

    at(x).cw2_load_pll_to_unroll = 0;    
    at(x).cw2_local_work_size = 64;//256;
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
    throw tinygemm_error("unrecognised x in get_n_elements_in_x_unroll");
  }
}


unsigned DerivedParams::get_stride(char x, bool pll_k, bool is_macro, unsigned workspace_type_) const{
  
  if (workspace_type_ == 0){
    return get_stride_cw0(x, pll_k);
  }
  
  else if (workspace_type_ == 1){
    return get_stride_cw1(x, pll_k);
  }
  
  else if (workspace_type_ == 2){
    return get_stride_cw2(x, pll_k, is_macro);
  }
  else throw tinygemm_error("unrecognised workspace_type in get_strinde in derivedparams");
}

unsigned DerivedParams::get_stride_cw0(char x, bool pll_k) const{
  return gg.coal_is_pll_k(x) == pll_k ? 1 : gg.get_ld(x); 
}

unsigned DerivedParams::get_stride_cw1(char x, bool pll_k) const{
  return gg.coal_is_pll_k(x) == pll_k ? 1 : at(x).cw1_target_ldx; 
}

unsigned DerivedParams::get_stride_cw2(char x, bool pll_k, bool is_macro) const{

  if (is_macro == false){
    return pll_k == true ? at(x).macro_tile_length : 1;
  }
  else{
    return pll_k == true ? at(x).macro_tile_length : gg.k;
  }
  
}



}
}
