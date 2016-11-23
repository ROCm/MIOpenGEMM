#include <string>

#include "betackernelutil.hpp"
#include "kernelsnips.hpp"

namespace tinygemm{
namespace betac{

const std::string cl_file_f32_path(PATH_BETAC_KERNEL_DIR + std::string("betackernel_f32.cl"));
const std::string cl_file_f64_path(PATH_BETAC_KERNEL_DIR + std::string("betackernel_f64.cl"));

std::string get_cl_file_path(char fchar){

  std::string cl_file_path;
  if (fchar == 'f'){
    cl_file_path = cl_file_f32_path;
  }
  else if (fchar == 'd'){
    cl_file_path = cl_file_f64_path;
  }
  else{
    std::string errm = "fchar has value `";
    errm += fchar;
    errm += "', which is not valid. It should be either `f' or `d'. This error is being throunw from set_betackernel_sizes, in betackernelutil.cpp";
    throw std::runtime_error(errm);
  }
  
  return cl_file_path;
}


void set_betackernel_sizes(char fchar, bool isColMajor, bool tC, unsigned m, unsigned n, unsigned & dim_coal, unsigned & dim_uncoal, size_t & betac_global_work_size, size_t & betac_local_work_size){
  if (isColMajor == true){
    dim_coal = tC ? n : m;
    dim_uncoal = tC ? m : n;
  }
  else{
    dim_coal = tC ? m : n;
    dim_uncoal = tC ? n : m;
  }
  
    
  auto betac_parms = kernelutil::get_integer_preprocessor_parameters(get_cl_file_path(fchar));       
  
  if (betac_parms.count("N_WORK_ITEMS_PER_GROUP") + betac_parms.count("WORK_PER_THREAD") != 2){
    throw std::runtime_error("It is required that both N_WORK_ITEMS_PER_GROUP and WORK_PER_THREAD are defined in the scaling kernel, something looks weird here.");
  }
  
  betac_local_work_size = betac_parms["N_WORK_ITEMS_PER_GROUP"];
  size_t work_per_thread = betac_parms["WORK_PER_THREAD"];    
  size_t n_betac_threads = dim_uncoal*(dim_coal/work_per_thread + ((dim_coal%work_per_thread) != 0));
  size_t number_of_betac_work_groups = (n_betac_threads / betac_local_work_size) + ((n_betac_threads % betac_local_work_size) != 0) ; 
  betac_global_work_size = number_of_betac_work_groups*betac_local_work_size;
}



}
}
