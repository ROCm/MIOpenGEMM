#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <utility>

#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{
namespace kernelutil{

  std::string
  get_as_single_string(std::string filename){
    std::ifstream input(filename);
    if(!input.good()){
      throw tinygemm_error( "Error attempting to open '" + filename + "', in kernelutil : :ge t_as_single_string.  ");
    }
    
    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
  }
  
  std::pair<std::map<std::string, unsigned>, std::map<std::string, std::string>> 
  get_all_preprocessor_parameters(const std::string & kernel_string){

    std::istringstream input(kernel_string);
    if(!input.good()){
      throw tinygemm_error( "Error converting parameter kernel_string into a std::istringstream (in kernelsnips.cpp). ");
    }
    
    
    std::string line;
    std::vector<std::string> frags;
    std::string hash_define_frag("#define");

    std::map<std::string, unsigned> integer_preprocessor_parameters;
    std::map<std::string, std::string> non_integer_preprocessor_parameters;
    
    while (std::getline(input, line).good() == true){
      if (line.size() >= hash_define_frag.size()){
        if (line.substr(0, hash_define_frag.size()).compare(hash_define_frag) == 0){
          /* we have a match to #define V */
          frags = stringutil::split(line);
          if (frags.size() < 3){
            std::string errm ("The following line in the .cl file : \n");
            errm += line;
            errm += "\nis unexpected, as it does not have enough fragments in it ";  
            throw tinygemm_error(errm);
          }
          
          else{
            std::string key = frags[1];
            //the cases where the value is not a unsigned should be caught here.
            if (key.compare("TFLOAT") == 0){
              non_integer_preprocessor_parameters[key] = frags[2];
            }
              
            
            else{
              std::string str_value = frags[2];
              unsigned value = std::stoi(str_value);
              if (std::to_string(value).compare(str_value) != 0){
                std::string errm("The following fragment cannot be converted into an int : \n");
                errm += str_value;
                errm += "\nin obtaining preprocessor parameters.\nIf you'd prefer it to go be passed a string, processing should mimic that of TFLOAT";
              }
              integer_preprocessor_parameters[key] = value;
            }
          }
        }
      }
    }
    
    return std::make_pair(integer_preprocessor_parameters, non_integer_preprocessor_parameters);
  }


  std::map<std::string, unsigned> get_integer_preprocessor_parameters(const std::string & kernel_string){
    auto all_preprocessor_parameters = get_all_preprocessor_parameters(kernel_string);
    return all_preprocessor_parameters.first;
  }



  std::string get_kernel_function_name(const std::string & kernel_string){
        
    std::istringstream input(kernel_string);
    if(!input.good()){
      throw tinygemm_error( "Error getting istringstream from parameter kernel_string in get_kernel_function_name (in kernelsnips.cpp) ");
    }
    
    
    std::string line;
    std::string function_name;
    while (std::getline(input, line).good() == true){
      std::string start = "__kern";
      if (line.size() > 10){
        if (line.substr(0, start.size()).compare(start) == 0){
          auto frags = stringutil::split(line);
          if (frags.size() >= 2){
            function_name = stringutil::split(frags[2],"(")[0];
            return function_name;
          }
        }
      }
    }
    throw tinygemm_error("Unable to determine the function name in .cl file");
  }


 void set_sizes_from_kernel_string(unsigned & macro_tile_width, unsigned & macro_tile_height, unsigned & n_workitems_per_workgroup, unsigned & n_work_items_per_c_elm, unsigned & does_beta_c_inc, const std::string & kernel_string){
    auto ipp = kernelutil::get_integer_preprocessor_parameters(kernel_string);
    macro_tile_width = ipp["MACRO_TILE_WIDTH"];
    macro_tile_height = ipp["MACRO_TILE_HEIGHT"];
    n_workitems_per_workgroup = ipp["N_WORK_ITEMS_PER_WORKGROUP"]; 
    n_work_items_per_c_elm = ipp["N_WORK_ITEMS_PER_C_ELM"];
    does_beta_c_inc = ipp["DOES_ALPHA_C_INC"]; //TODO : `ALPHA' in kernel headers should be BETA. 
  }


}} //namespace





