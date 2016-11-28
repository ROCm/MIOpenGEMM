#include "makekernelsource.hpp"
#include "hyperparams.hpp"
#include "defaultoutpath.hpp"
#include "tinygemmerror.hpp"


#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

namespace tinygemm{
namespace mkkern{
  
/* returns 0 if the kernel was succesfully written. If the hyperparameters don't gel, a kernel will not be written, and a non-zero int will be returned */
int make_kernel_via_python(std::string dir_name, std::string t_float, std::map<std::string, unsigned> all_int_parms, std::string kernelname, bool verbose_report){

  
  std::string parameter_string(" --dir_name ");
  parameter_string += dir_name;
  
  parameter_string += "  --kernelname  ";
  parameter_string += kernelname;
  
  
  parameter_string += " --t_float ";
  parameter_string += t_float;
  
  

  for (auto x : all_int_parms){
    if (std::count(hyperparams::all_int_param_names.cbegin(), hyperparams::all_int_param_names.cend(), x.first) == 0){
      std::string errm("The received parameter, `");
      errm += x.first;
      errm += "' does not appear to be a valid parameter. ";
      throw tinygemm_error(errm);
    }
  }
  
  for (std::string x : hyperparams::all_int_param_names){
    if (all_int_parms.count(x) == 0){
      std::string errm("The parameter `");
      errm += x;
      errm += "' appears to be missing in make_kernel_via_parameters";
      throw tinygemm_error(errm);
    }
    
    else{
      parameter_string += "  --";
      parameter_string += x;
      parameter_string += " ";
      parameter_string += std::to_string(all_int_parms[x]);
    }
  }

  //std::cout << PATH_MAKE_KERNEL_CMDL_PY << std::endl;
  std::string syscall(PATH_MAKE_KERNEL_CMDL_PY );
  syscall += " ";
  syscall += parameter_string;
  
  
  //bool verbose_report = true;
  if (verbose_report == true){
    syscall += " 0>>/dev/null";
    /* print want make_kernel does, good and bad */ 
  }
  else{
    syscall += " 2>>/dev/null";
  }
  
  //TODO : pipe raised python errors elsewhere, in case an important one is thrown.    
   int success = std::system(syscall.c_str());  
  
  return success;
}

}} //namespace


 
