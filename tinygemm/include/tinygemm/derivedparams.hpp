#ifndef TINYGEMM_DERIVEDPARAMS_HPP
#define TINYGEMM_DERIVEDPARAMS_HPP

#include <vector>
#include <string>
#include <map>

namespace tinygemm{
namespace derivedparams{

static std::vector<std::string> all_derived_unsigned_param_names {
"split_on_k",
"does_alpha_c_inc",
"macro_tile_area",
"micro_tile_area",
"n_workitems_per_workgroup",
"macro_tile_height_and_pad",
"macro_tile_width_and_pad",
"n_elements_in_a_unroll",
"n_elements_in_b_unroll",
"n_elements_in_padded_a_unroll",
"n_elements_in_padded_b_unroll",
"n_elements_of_a_to_load_per_workitem",
"n_elements_of_b_to_load_per_workitem",
"n_micro_tiles_vertically",
"n_micro_tiles_horizontally",
"micro_a_tile_pll_unroll" ,
"micro_a_tile_perp_unroll", 
"n_micro_a_tiles_pll_unroll",  
"micro_b_tile_pll_unroll",
"micro_b_tile_perp_unroll",
"n_micro_b_tiles_pll_unroll" };


static std::vector<std::string> all_derived_string_param_names
{/* */
"strided_i_vertical",
/* */
"strided_i_horizontal",
/*the int type for atomics */
"infa",
/* the function to use for atomic ints */
"fati",
/* */
"effective_k", 
/* */
"alpha_scaled",
/* pragma unroll string ("#pragma unroll\n" or "") */
"pragma_unroll_string",
/* currently one of "float" and "double", set from float_size */
"t_float" };

}
}


#endif
