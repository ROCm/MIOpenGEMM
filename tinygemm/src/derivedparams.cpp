#include <tinygemm/derivedparams.hpp>
#include <tinygemm/mapkeycheck.hpp>
#include <tinygemm/tiling.hpp>


namespace tinygemm{
namespace derivedparams{




  
unsigned get_target(unsigned grid_size, unsigned above_distance, unsigned x){
  unsigned to_grid_line = (x - above_distance) / grid_size  + ((x - above_distance) % grid_size != 0);
  return grid_size * to_grid_line + above_distance;
} 

/* TODO : this 3 is ~ arbitrary : make hyper parameter ! */
void DerivedParams::reset_acw1_params(const tinygemm::TinyGemmGeometry & gg){
  
  /* what lda would be if no `padding' */
  acw1_smallest_possible_lda = gg.tA == gg.isColMajor ? gg.k : gg.m;
  /* smallest x s.t. x >= acw1_smallest_possible_lda and x = n*16 + 3. */ 
  acw1_target_lda = get_target(16, 3, acw1_smallest_possible_lda);
  
}

void DerivedParams::reset_nf_params(const tinygemm::TinyGemmGeometry & gg, const hyperparams::HyperParams & hp){
  nf_k_normal_form = hp.unroll*(gg.k / hp.unroll + ((gg.k % hp.unroll) != 0));
}

/* TODO : this 0 is ~ arbitrary : make hyper parameter ! */
void DerivedParams::reset_bcw1_params(const tinygemm::TinyGemmGeometry & gg, const hyperparams::HyperParams & hp){

  if (hp.a_copy_workspace == 1 && acw1_target_lda == uninitialised_unsigned){
    throw tinygemm_error("make sure reset_acw1_params is called before reset_bcw1_params, we need that acw1_target_lda be set here in derivedparams reset of bcw1");
  }
  
  /* what ldb would be if no `padding' */
  bcw1_smallest_possible_ldb =  gg.tB == gg.isColMajor ? gg.n : gg.k;
  /* smallest x s.t. x >= bcw1_smallest_possible_ldb and x = n*16 + 6. */ 
  bcw1_target_ldb =  get_target(16, 6, bcw1_smallest_possible_ldb);


  bcw1_global_offset_b = hp.a_copy_workspace == 1 ? get_target_ld('a')*gg.get_uncoal('a') : 0;
}
  
void DerivedParams::reset_ga3_params(const hyperparams::HyperParams & hp){
  
  if (split_on_k == 1){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.n_target_active_workgroups) / static_cast<double>(hp.n_work_items_per_c_elm))));
  }
  else if (split_on_k == 0){
    ga3_super_column_width = 
    static_cast<unsigned>(std::floor(std::sqrt(static_cast<double>(hp.n_target_active_workgroups))));
  }
  else{
    throw tinygemm_error("split_on_k is neither 0 nor 1, how can this be? Logic error in append_super_column_width_defn");
  }  
  ga3_last_super_column_width = n_groups_horizontally % ga3_super_column_width;
}


std::tuple<bool, std::string>
get_deriveability(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg){
  DerivedParams dp("uninitialised");
  return dp.set_fragile(hp, gg);
}

DerivedParams::DerivedParams(std::string s){
  if (s.compare("uninitialised") != 0){
    throw tinygemm_error("the only string with which a DerivedParams object can be initialised is `uninitialised'");
  }
}


std::tuple<bool, std::string>
DerivedParams::set_fragile(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg){

  macro_tile_area = hp.macro_tile_width * hp.macro_tile_height;
  micro_tile_area = hp.micro_tile_width * hp.micro_tile_height;
  main_n_work_items_per_workgroup = macro_tile_area / micro_tile_area;
  n_elements_in_a_unroll = hp.macro_tile_height * hp.unroll;
  n_elements_in_b_unroll = hp.macro_tile_width * hp.unroll;
  n_elements_of_a_to_load_per_workitem = n_elements_in_a_unroll / main_n_work_items_per_workgroup;
  n_elements_of_b_to_load_per_workitem = n_elements_in_b_unroll / main_n_work_items_per_workgroup;  
  
  std::stringstream set_status_ss;  
  
  /* check -1 : enough workspace memory */
  unsigned required_workspace = 0;
  if (hp.a_copy_workspace == 1){
    reset_acw1_params(gg);
    required_workspace += acw1_target_lda*gg.get_uncoal('a');
  }

  if (hp.b_copy_workspace == 1){
    reset_bcw1_params(gg, hp);
    required_workspace += bcw1_target_ldb*gg.get_uncoal('b');
  }

  if (gg.workspace_size < required_workspace){
    set_status_ss << "gg.workspace_size ( " << gg.workspace_size << " ) is less then the required workspace ( " << required_workspace << " ) "; 
  }
  
  
  /* check 0 : macro tile not too large */
  if (gg.m < hp.macro_tile_height){
    set_status_ss << "m < macro_tile_height, not considering this kernel\n";
  }
  if (gg.n < hp.macro_tile_width){
    set_status_ss << "n < macro_tile_width,  not considering this kernel\n";
  }
  if (set_status_ss.str() != ""){
    return std::make_tuple(false, set_status_ss.str()); 
  }
  
  /* check 1 : n_work_items_per_workgroup divides n_elements_in_a_unroll and n_elements_in_b_unroll  */
  if (n_elements_in_a_unroll % main_n_work_items_per_workgroup != 0){
    set_status_ss << "this is not supported:\nn_work_items_per_workgroup (" << main_n_work_items_per_workgroup << ") is not a factor of n_elements_in_" <<  "a" << "_UNROLL (" << n_elements_in_a_unroll << "). \nConsider rounding unroll up. \n";
  }
  if (n_elements_in_b_unroll % main_n_work_items_per_workgroup != 0){
    set_status_ss << "this is not supported:\nn_work_items_per_workgroup (" << main_n_work_items_per_workgroup << ") is not a factor of n_elements_in_" <<  "b" << "_UNROLL (" << n_elements_in_b_unroll << "). \nConsider rounding unroll up. \n";
  }
  if (set_status_ss.str() != ""){
    return std::make_tuple(false, set_status_ss.str()); 
  }
  
  /* check 2 : tilability */
  auto tup1 = hp.work_item_load_a_pll_to_unroll == 0 ? 
  tiling::get_tileability(hp.macro_tile_height, hp.unroll, n_elements_of_a_to_load_per_workitem):
  tiling::get_tileability(hp.unroll, hp.macro_tile_height, n_elements_of_a_to_load_per_workitem);
  
  auto tup2 = hp.work_item_load_b_pll_to_unroll == 0 ?
  tiling::get_tileability(hp.macro_tile_width, hp.unroll, n_elements_of_b_to_load_per_workitem):
  tiling::get_tileability(hp.unroll, hp.macro_tile_width, n_elements_of_b_to_load_per_workitem);

  for (auto & tup : {tup1, tup2}){
    if (std::get<0>(tup) == false){
      return tup;
    }
  }
  
  /* ran the gauntlet, returning deriveable is true */
  return std::make_tuple(true, "");
}


DerivedParams::DerivedParams(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg){

  auto tup = set_fragile(hp, gg);

  if (std::get<0>(tup) == false){
    throw tinygemm_error("Failure to construct DerivedParams. Problem caught in set_fragile. It is recommended to run use function derivable to check that a valid DerivedParams can be constructed. The message returned in set_fragile is :  " + std::get<1>(tup));
  }
  
  /* do the tiling */
  if (hp.work_item_load_a_pll_to_unroll == 0){
    tiling::set_tile_dimensions(micro_a_tile_perp_unroll, micro_a_tile_pll_unroll, hp.macro_tile_height, hp.unroll, n_elements_of_a_to_load_per_workitem); 
  }
  
  else{
    tiling::set_tile_dimensions(micro_a_tile_pll_unroll, micro_a_tile_perp_unroll, hp.unroll, hp.macro_tile_height, n_elements_of_a_to_load_per_workitem);
  }
  
  if (hp.work_item_load_b_pll_to_unroll == 0){
    tiling::set_tile_dimensions(micro_b_tile_perp_unroll, micro_b_tile_pll_unroll, hp.macro_tile_width, hp.unroll, n_elements_of_b_to_load_per_workitem); 
  }
  else{
    tiling::set_tile_dimensions(micro_b_tile_pll_unroll, micro_b_tile_perp_unroll, hp.unroll, hp.macro_tile_width, n_elements_of_b_to_load_per_workitem);
  }
    
    
  split_on_k = hp.n_work_items_per_c_elm == 1 ? 0 : 1;
  does_beta_c_inc = split_on_k == 1 ? 0 : 1; 
  
  strided_i_vertical = hp.c_micro_tiles_interwoven == 0 ? "i" : "i*N_MICRO_TILES_VERTICALLY";
  strided_i_horizontal = hp.c_micro_tiles_interwoven == 0 ? "i" : "i*N_MICRO_TILES_HORIZONTALLY";
  
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
  
  effective_k_varies_string = hp.normal_form == 1 ? "__K_NORMAL_FORM__" : (hp.unroll_for_offset == 0 ? "__K__" : "k_plus_offset");
  
  alpha_scaled = hp.c_micro_tiles_interwoven == 0 ? "alpha*rC[row][col]" : "alpha*rC[row/N_MICRO_TILES_VERTICALLY][col/N_MICRO_TILES_HORIZONTALLY]";
  
  t_float = gg.derived.float_size_bits == 32 ? "float" : "double";
  
  preshift_bottommost_tile_height = 1 + (gg.m - 1) % hp.macro_tile_height;
  preshift_rightmost_tile_width = 1 + (gg.n - 1) % hp.macro_tile_width;
  n_groups_vertically = gg.m / hp.macro_tile_height + (preshift_bottommost_tile_height != hp.macro_tile_height);
  n_groups_horizontally = gg.n / hp.macro_tile_width + (preshift_rightmost_tile_width != hp.macro_tile_width);
  macro_tile_height_and_pad = hp.macro_tile_height + hp.pad;
  macro_tile_width_and_pad = hp.macro_tile_width + hp.pad;
  n_elements_in_padded_a_unroll = macro_tile_height_and_pad * hp.unroll;
  n_elements_in_padded_b_unroll = macro_tile_width_and_pad * hp.unroll;
  n_micro_tiles_vertically = hp.macro_tile_height / hp.micro_tile_height;
  n_micro_tiles_horizontally = hp.macro_tile_width / hp.micro_tile_width ;
  main_n_work_groups = hp.n_work_items_per_c_elm * ((gg.m/hp.macro_tile_height) + (gg.m%hp.macro_tile_height != 0)) * ((gg.n/hp.macro_tile_width) + (gg.n%hp.macro_tile_width != 0));
  main_global_work_size = main_n_work_groups * main_n_work_items_per_workgroup;
  n_micro_a_tiles_pll_unroll = hp.unroll / micro_a_tile_pll_unroll;
  n_micro_b_tiles_pll_unroll = hp.unroll / micro_b_tile_pll_unroll;

  needs_final_fractional_unroll = (hp.unroll_for_offset == 1 || (hp.normal_form == 0 && gg.k%hp.unroll != 0)) ? 1 : 0;
    
  if (hp.group_allocation == 3){
    reset_ga3_params(hp);//hp, n_groups_horizontally, split_on_k);
  }
  
  if (hp.normal_form == 1){
    reset_nf_params(gg, hp);
  }
  
  if (hp.normal_form == 0){
    effective_lda = hp.a_copy_workspace == 0 ? gg.lda : acw1_target_lda;
    effective_ldb = hp.b_copy_workspace == 0 ? gg.ldb : bcw1_target_ldb;
  }
}


unsigned DerivedParams::get_target_ld(char c) const{
  if (c == 'a'){
    return acw1_target_lda;
  }
  else if (c == 'b'){
    return bcw1_target_ldb;
  }
  else{
    throw tinygemm_error("unrecognised char in get_target_ld(char c)");
  }  
}


}
}
