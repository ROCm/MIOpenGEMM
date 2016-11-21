#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "tinygemmsolution.hpp"
#include "sizingup.hpp"
#include "betackernelutil.hpp"


namespace tinygemm{

std::string get_errm(std::string x){
  std::string errm("\nThis error is being thrown while trying to set the kernel work size. It appears that the parameter `");
  errm += x;
  errm += "' is not available in the member map, integral_hyperparams. This is unexpected, and requires further investigation.\n"; 
  return errm;
}


std::map<std::string, size_t> TinyGemmSolution::get_main_kernel_worksize_params(unsigned m, unsigned n){

  /* to populate and return */
  std::map<std::string, size_t> worksize_params;
  
  /* performing a check for map keys */
  std::vector<std::string> integral_params_needed {"n_work_items_per_c_elm", "macro_tile_height", "macro_tile_width", "micro_tile_height", "micro_tile_width"}; //, "n_workitems_per_workgroup"};
  for (auto & x : integral_params_needed){
    if (allparams.count(x) == 0){
  
      
      //std::cout << "allparams:\n";
      //for (auto & y : allparams){
        //std::cout << y.first  << " " << y.second << std::endl;
      //}
    
      //zzzzzzzzzzzzzzzzzzzzzzzzzzzzz
      throw std::runtime_error(get_errm(x));
    }
  }
  
  unsigned n_workitems_per_workgroup = ( allparams.at("macro_tile_height")*allparams.at("macro_tile_width") ) / ( allparams.at("micro_tile_height")*allparams.at("micro_tile_width") );
  
  
  /* outsource the calculation */
  size_t n_work_groups, local_work_size, global_work_size;
  sizingup::set_workforce(n_work_groups, local_work_size, global_work_size, m, n, allparams["n_work_items_per_c_elm"], allparams["macro_tile_height"], allparams["macro_tile_width"], n_workitems_per_workgroup);
  
  /* populate and return */
  worksize_params["n_work_groups"] = n_work_groups;
  worksize_params["local_work_size"] = local_work_size;
  worksize_params["global_work_size"] = global_work_size;
  
  return worksize_params;

}


std::map<std::string, size_t> TinyGemmSolution::get_betac_kernel_worksize_params(unsigned m, unsigned n){

  std::map<std::string, size_t> betac_worksize_params;
  
  std::vector<std::string> integral_params_needed {"is_col_major", "c_transposed"};
  for (auto & x : integral_params_needed){
    if (allparams.count(x) == 0){
      throw std::runtime_error(get_errm(x));
    }
  }
 
  unsigned dim_coal, dim_uncoal;
  size_t betac_global_work_size, betac_local_work_size;

  if (floattype != 'd' && floattype != 'f'){
    std::string errm("Just checking that floattype is either 'd' or 'f' in get_betac_kernel_worksize_params in tinygemmsolution.cpp, and it is not, it is `");
    errm += floattype;
    errm += "'. This is strange (14/11/2016, we only support 32 and 64 bit floats)";
    throw std::runtime_error(errm);
  }
  
  betac::set_betackernel_sizes(floattype, allparams["is_col_major"], allparams["c_transposed"], m, n, dim_coal, dim_uncoal, betac_global_work_size, betac_local_work_size);  

  betac_worksize_params["dim_coal"] = dim_coal;
  betac_worksize_params["dim_uncoal"] = dim_coal;
  betac_worksize_params["global_work_size"] = betac_global_work_size;
  betac_worksize_params["local_work_size"] = betac_local_work_size;    

  return betac_worksize_params;

}

std::string TinyGemmSolution::get_hyper_param_string(){

  std::stringstream ss;
  ss << "Y" << allparams["macro_tile_height"] <<  "_X" << allparams["macro_tile_width"] << "_y" << allparams["micro_tile_height"] << "_x" << allparams["micro_tile_width"] << "_U" << allparams["unroll"] << "_P" << allparams["pad"] << "_GA" << allparams["group_allocation"] << "_APLU" << allparams["work_item_load_a_pll_to_unroll"] << "_BPLU" << allparams["work_item_load_b_pll_to_unroll"] << "_PU" << allparams["unroll_pragma"] << "_LIW" << allparams["load_to_lds_interwoven"] << "_MIW" << allparams["c_micro_tiles_interwoven"]  << "_ET" << 1 << "_ICE" << allparams["n_work_items_per_c_elm"] << "_UFO" << allparams.at("unroll_for_offset");
  
  return ss.str();
}


} //namespace
