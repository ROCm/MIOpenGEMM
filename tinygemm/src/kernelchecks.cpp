#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <tinygemm/kernelsnips.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{
namespace kernelutil{

void run_preprocessor_parameter_tests(std::map<std::string, unsigned> ipps, std::map<std::string, std::string> spps,
bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring){


  bool isError = false;
  std::string errm ("An issue (or several) emerged in checking the preprocessor flags in run_preproprocessor_parameter_tests.");
  errm += ". \n";

  std::vector<std::string> all_non_integer_preprocessor_parameters = {"TFLOAT"};
                
  
  std::vector<std::string> all_integer_preprocessor_parameters = {"IS_COL_MAJOR", "A_TRANSPOSED", "B_TRANSPOSED", "C_TRANSPOSED", "DOES_ALPHA_C_INC", "DOES_BETA_A_B_INC", "MICRO_TILE_WIDTH", "MICRO_TILE_HEIGHT", "MACRO_TILE_WIDTH", "MACRO_TILE_HEIGHT", "UNROLL", "PAD", "EDGETRICK", "N_WORK_ITEMS_PER_C_ELM", "GROUP_ALLOCATION", "WORK_ITEM_LOAD_A_PLL_TO_UNROLL", "WORK_ITEM_LOAD_B_PLL_TO_UNROLL", "LOAD_TO_LDS_INTERWOVEN", "C_MICRO_TILES_INTERWOVEN", "PRAGMA_UNROLL_FORLOOPS", "N_PREFETCH_FOR_REGISTER_LOAD", "N_PREFETCH_FOR_LDS_LOAD", "MACRO_TILE_AREA", "MICRO_TILE_AREA", "N_WORK_ITEMS_PER_WORKGROUP", "MACRO_TILE_HEIGHT_AND_PAD", "MACRO_TILE_WIDTH_AND_PAD", "N_ELEMENTS_IN_A_UNROLL", "N_ELEMENTS_IN_B_UNROLL", "N_ELEMENTS_IN_PADDED_A_UNROLL", "N_ELEMENTS_IN_PADDED_B_UNROLL", "N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM", "N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM", "N_MICRO_TILES_VERTICALLY", "N_MICRO_TILES_HORIZONTALLY", "MICRO_A_TILE_PLL_UNROLL", "MICRO_A_TILE_PERP_UNROLL", "N_MICRO_A_TILES_PLL_UNROLL", "MICRO_B_TILE_PLL_UNROLL", "MICRO_B_TILE_PERP_UNROLL", "N_MICRO_B_TILES_PLL_UNROLL", "M_FACTOR", "N_FACTOR", "UNROLL_FOR_OFFSET"};
  
  
      
      
      
  std::vector<std::string> all_preprocessor_parameters;
  all_preprocessor_parameters.insert(all_preprocessor_parameters.end(), all_integer_preprocessor_parameters.begin(), all_integer_preprocessor_parameters.end());
  all_preprocessor_parameters.insert(all_preprocessor_parameters.end(), all_non_integer_preprocessor_parameters.begin(), all_non_integer_preprocessor_parameters.end());
  
  for (auto & x : all_preprocessor_parameters){
    if (ipps.count(x) + spps.count(x) == 0 ){
      errm += "The preprocessor paramater ";
      errm += x;
      errm += " appears to be undefined\n";
      isError = true;
    }
  }
  
  auto increrm = [&errm, &isError](std::string key, std::string mapname){
    errm += "key "; 
    errm += key; 
    errm += " not a key of ";
    errm += mapname;
    isError = true;
  };
  
  auto ipps_with_check = [&increrm, &ipps, &isError, &errm](std::string key){
    if (ipps.count(key) == 0){
      increrm(key, "ipps");
    }    
    return ipps[key];
  };

  auto spps_with_check = [&increrm, &spps, &isError, &errm](std::string key){
    if (spps.count(key) == 0){
      increrm(key, "spps");
    }
    return spps[key];
  };
  
  if (spps_with_check("TFLOAT").compare(floatstring) != 0){
    errm += "Incompatible types. The kernel expects \n";
    errm += spps_with_check("TFLOAT");
    errm += " \nwhile the data type `here' is \n";
    errm += floatstring;
    errm += "\n";
    isError = true;
  }

  if (tA != ipps_with_check("A_TRANSPOSED")){
    errm += "tA != A_TRANSPOSED\n";
    isError = true;
  }

  if (tB != ipps_with_check("B_TRANSPOSED")){
    errm += "tB != B_TRANSPOSED\n";
    isError = true;
  }


  if (tC != ipps_with_check("C_TRANSPOSED")){
    errm += "tC = ";
    errm += tC;
    errm += "\t C_TRANSPOSED = ";
    errm += ipps_with_check("C_TRANSPOSED");
    errm += "   ==>   tC != C_TRANSPOSED\n";
    isError = true;
  } 

  if (isColMajor != ipps_with_check("IS_COL_MAJOR")){
    errm += "isColMajor != IS_COL_MAJOR\n";
    isError = true;
  }
  
  if (ipps_with_check("MACRO_TILE_WIDTH")*ipps_with_check("MACRO_TILE_HEIGHT") != ipps_with_check("MACRO_TILE_AREA")){
    errm += " MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT != MACRO_TILE_AREA\n";
    isError = true;
  }
  
  if (ipps_with_check("MACRO_TILE_HEIGHT_AND_PAD") != ipps_with_check("MACRO_TILE_HEIGHT") + ipps_with_check("PAD")){
    errm += "MACRO_TILE_HEIGHT_AND_PAD != MACRO_TILE_HEIGHT + PAD\n";
    isError = true;
  }

  if (ipps_with_check("MACRO_TILE_WIDTH_AND_PAD") != ipps_with_check("MACRO_TILE_WIDTH") + ipps_with_check("PAD")){
    errm += "MACRO_TILE_WIDTH_AND_PAD != MACRO_TILE_WIDTH + PAD\n";
    isError = true;
  }
  
  //if (k % ipps_with_check("UNROLL") != 0){
    //errm += "UNROLL does not divide k. For the time being, this is considered a `bug'\n";
    //isError = true;
  //}
  
  if (ipps_with_check("N_WORK_ITEMS_PER_WORKGROUP") != ipps_with_check("MACRO_TILE_AREA") / ipps_with_check("MICRO_TILE_AREA")){
    errm += "N_WORK_ITEMS_PER_WORKGROUP != MACRO_TILE_AREA / MICRO_TILE_AREA. When splitting in k is allowed, this may not be an error, but for the moment it is\n";
    isError = true;
  }
  
  if (ipps_with_check("MICRO_TILE_AREA") != ipps_with_check("MICRO_TILE_WIDTH") * ipps_with_check("MICRO_TILE_HEIGHT")){
    errm += "MICRO_TILE_AREA != MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT\n";
    isError = true;
  }
  
  if (ipps_with_check("N_ELEMENTS_IN_A_UNROLL") != ipps_with_check("MACRO_TILE_HEIGHT") * ipps_with_check("UNROLL")){
    errm += "N_ELEMENTS_IN_A_UNROLL != MACRO_TILE_HEIGHT * UNROLL\n";
    isError = true;
  }

  if (ipps_with_check("N_ELEMENTS_IN_B_UNROLL") != ipps_with_check("MACRO_TILE_WIDTH") * ipps_with_check("UNROLL")){
    errm += "N_ELEMENTS_IN_B_UNROLL != MACRO_TILE_WIDTH * UNROLL\n";
    isError = true;
  }
  
  if (ipps_with_check("N_ELEMENTS_IN_PADDED_A_UNROLL") != ipps_with_check("MACRO_TILE_HEIGHT_AND_PAD") * ipps_with_check("UNROLL")){
    errm += "N_ELEMENTS_IN_PADDED_A_UNROLL != MACRO_TILE_HEIGHT_AND_PAD * UNROLL\n";
    isError = true;
  }

  if (ipps_with_check("N_ELEMENTS_IN_PADDED_B_UNROLL") != ipps_with_check("MACRO_TILE_WIDTH_AND_PAD") * ipps_with_check("UNROLL")){
    errm += "N_ELEMENTS_IN_PADDED_B_UNROLL != MACRO_TILE_WIDTH_AND_PAD * UNROLL\n";
    isError = true;
  }
  
  
  if (ipps_with_check("N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM") != ipps_with_check("N_ELEMENTS_IN_A_UNROLL") / ipps_with_check("N_WORK_ITEMS_PER_WORKGROUP")){
    errm += "N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM != N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
    isError = true;      
  }

  if (ipps_with_check("N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM") != ipps_with_check("N_ELEMENTS_IN_B_UNROLL") / ipps_with_check("N_WORK_ITEMS_PER_WORKGROUP")){
    errm += "N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM != N_ELEMENTS_IN_B_UNROLL / N_WORK_ITEMS_PER_WORKGROUP\n";
    isError = true;      
  }
  
  if (ipps_with_check("N_MICRO_TILES_VERTICALLY") !=  ipps_with_check("MACRO_TILE_HEIGHT") / ipps_with_check("MICRO_TILE_HEIGHT")){
    errm += "N_MICRO_TILES_VERTICALLY != MACRO_TILE_HEIGHT / MICRO_TILE_HEIGHT\n";
    isError = true;
  }
  

  if (ipps_with_check("N_MICRO_TILES_HORIZONTALLY") !=  ipps_with_check("MACRO_TILE_WIDTH") / ipps_with_check("MICRO_TILE_WIDTH")){
    errm += "N_MICRO_TILES_HORIZONTALLY != MACRO_TILE_WIDTH / MICRO_TILE_WIDTH\n";
    isError = true;
  }
  
  if (ipps_with_check("MICRO_A_TILE_PLL_UNROLL") * ipps_with_check("MICRO_A_TILE_PERP_UNROLL") != ipps_with_check("N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM")){
    errm += "MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL != N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM\n";
    isError = true;
  }
  
  if (ipps_with_check("N_MICRO_A_TILES_PLL_UNROLL") !=  ipps_with_check("UNROLL") / ipps_with_check("MICRO_A_TILE_PLL_UNROLL")){
    errm += "N_MICRO_A_TILES_PLL_UNROLL != UNROLL / MICRO_A_TILE_PLL_UNROLL\n";
    isError = true;
  }

  if (ipps_with_check("MICRO_B_TILE_PLL_UNROLL") * ipps_with_check("MICRO_B_TILE_PERP_UNROLL") != ipps_with_check("N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM")){
    errm += "MICRO_B_TILE_PLL_UNROLL * MICRO_B_TILE_PERP_UNROLL != N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM\n";
    isError = true;
  }
  
  if (ipps_with_check("N_MICRO_B_TILES_PLL_UNROLL") !=  ipps_with_check("UNROLL") / ipps_with_check("MICRO_B_TILE_PLL_UNROLL")){
    errm += "N_MICRO_B_TILES_PLL_UNROLL != UNROLL / MICRO_B_TILE_PLL_UNROLL\n";
    isError = true;
  }

  if (ipps_with_check("EDGETRICK") == 0 && m%ipps_with_check("MACRO_TILE_HEIGHT") != 0 ){
    errm += "EDGETRICK = 0 and m % MACRO_TILE_HEIGHT != 0\n";
    isError = true;
  }

  if (ipps_with_check("EDGETRICK") == 0 && n%ipps_with_check("MACRO_TILE_WIDTH") != 0 ){
    errm += "EDGETRICK = 0 and n % MACRO_TILE_WIDTH != 0\n";
    isError = true;
  }


  if (m%ipps_with_check("M_FACTOR") != 0 ){
    errm += "m % M_FACTOR != 0\n";
    isError = true;
  }

  if (n%ipps_with_check("N_FACTOR") != 0 ){
    errm += "n % N_FACTOR != 0\n";
    isError = true;
  }

  if (ipps_with_check("EDGETRICK") == 1 && ipps_with_check("M_FACTOR") != 1 ){
    errm += "EDGETRICK = 1 and M_FACTOR != 1\n";
    isError = true;
  }

  if (ipps_with_check("EDGETRICK") == 1 && ipps_with_check("N_FACTOR") != 1 ){
    errm += "EDGETRICK = 1 and N_FACTOR != 1\n";
    isError = true;
  }
  
  if (ipps_with_check("N_WORK_ITEMS_PER_C_ELM") > 1){
    if (ipps_with_check("UNROLL")*ipps_with_check("N_WORK_ITEMS_PER_C_ELM") != ipps_with_check("G_UNROLL")){
      errm += "UNROLL*N_WORK_ITEMS_PER_C_ELM  != G_UNROLL\n";
      isError = true;        
    }
  }
  
  if (m < ipps_with_check("MACRO_TILE_HEIGHT")){
    errm += "m < MACRO_TILE_HEIGHT\n";
    isError = true;
  }

  if (n < ipps_with_check("MACRO_TILE_WIDTH")){
    errm += "m < MACRO_TILE_WIDTH\n";
    isError = true;
  }

  if (ipps_with_check("N_PREFETCH_FOR_LDS_LOAD") > 3){
    errm += "N_PREFETCH_FOR_LDS_LOAD > 3 : NOT ALLOWED. should be 0. maaaaybe 1. but probably 0. \n";
    isError = true;
  }
  
  if (ipps_with_check("DOES_BETA_A_B_INC") != 1){
    errm += "DOES_BETA_A_B_INC != 1. There are currently (27/10/2016) no kernels which do not do the alpha*a*b, this looks weird \n";
    isError = true;      
  }
  
  if (ipps_with_check("DOES_ALPHA_C_INC") != 1 && ipps_with_check("N_WORK_ITEMS_PER_C_ELM") == 1){
    errm += "DOES_ALPHA_C_INC != 1 and N_WORK_ITEMS_PER_C_ELM == 1. This looks weird, currently all kernels with 1 work item per c element do the alpha scaling \n";
    isError = true;
  }

  if (ipps_with_check("DOES_ALPHA_C_INC") == 1 && ipps_with_check("N_WORK_ITEMS_PER_C_ELM") != 1){
    errm += "DOES_ALPHA_C_INC == 1 and N_WORK_ITEMS_PER_C_ELM != 1. This looks weird, currently no kernels with more than 1 work item per c element do not do the alpha scaling \n";
    isError = true;
  }        
  
  if (isError == true){
    throw tinygemm_error(errm);
  }
}


void check_gpu_kernel_filename(std::string kernel_filename){
  std::ifstream input(kernel_filename);
  if(!input.good()){
    std::string errm("This error is being thrown from check_gpu_kernel_filename, in kernelchecks.cpp. The kernel_filename passed in is ` ");
    errm +=  kernel_filename;
    errm += "', which for some reason cannot be opened.  ";
    throw tinygemm_error(errm);
  }
}


/* check that the files are readable */
void check_gpu_kernel_filenames(const std::vector<std::vector<std::string>> & gpu_kernel_filenames){
  for (auto & v : gpu_kernel_filenames){
    if (v.size() != 1){
      std::string errm("The size of this element of gpu_kernel_filenames is ");
      errm += std::to_string(v.size());
      errm += ".\nCurrently, we can only handle 1 kernel file. \nCurrently, there is no need to support multiple kernel files. ";
      throw tinygemm_error(errm);
    }
    for (auto & kernel_filename : v){
      check_gpu_kernel_filename(kernel_filename);
    }
  }
}


void check_gpu_kernel_preprocessor_parameters(std::string kernel_filename, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring){
  auto all_preprocessor_parameters = kernelutil::get_all_preprocessor_parameters(kernel_filename);
  auto integer_preprocessor_parameters = all_preprocessor_parameters.first;
  auto non_integer_preprocessor_parameters = all_preprocessor_parameters.second;
  /* we have now extracted all of the preprocessor parameters from the file, let's check them.*/
  kernelutil::run_preprocessor_parameter_tests(integer_preprocessor_parameters, non_integer_preprocessor_parameters, tA, tB, tC, isColMajor, m, n, floatstring);

}

void check_gpu_kernels_preprocessor_parameters(const std::vector<std::vector<std::string>> & gpu_kernel_filenames, bool tA, bool tB, bool tC, bool isColMajor, unsigned m, unsigned n, std::string floatstring){
  for (auto & v : gpu_kernel_filenames){
    for (auto & kernel_filename : v){
      check_gpu_kernel_preprocessor_parameters(kernel_filename, tA, tB, tC, isColMajor, m, n, floatstring);
    }
  }
}


}} //namespace
