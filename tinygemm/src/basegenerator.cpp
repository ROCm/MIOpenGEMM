#include <tinygemm/basegenerator.hpp>

#include <sstream>
#include <chrono>
#include <sstream>

namespace tinygemm{
namespace basegen{

      
BaseGenerator::BaseGenerator(const tinygemm::hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_): hp(hp_), gg(gg_), dp(dp_), type(type_), kernelname("tg_" + type_) {

  //kernelname = get_generic_kernelname(type);
}
  

void BaseGenerator::append_parameter_list_from_usage(std::stringstream & ss){
  
  char first_char = '\n';
  
  ss << "\n(";
  if (uses_a == true){
    ss << first_char << "__global const TFLOAT * restrict a, \n" << "const unsigned a_offset";
    first_char = ',';
  }

  if (uses_b == true){
    ss << first_char << "\n__global const TFLOAT * restrict b, \n" << "const unsigned b_offset";
    first_char = ',';
  }
  
  if (uses_c == true){
    ss << first_char << "\n__global TFLOAT       *          c, \n" << "const unsigned c_offset";
    first_char = ',';
  }

  if (uses_workspace == true){
    
    //if using c, workspace is const. this is a bit hacky, might have a kernel which uses c and modifies w too. 
    if (uses_c == true){
      ss << first_char << "\n__global const TFLOAT * restrict w,\n";
    }
    else{
      ss << first_char << "\n__global TFLOAT * restrict w,\n";
    }
    ss << "const unsigned w_offset";
    first_char = ',';
  }


  if (uses_alpha == true){
    ss << first_char << "\nconst TFLOAT alpha";
    first_char = ',';
  }
  
  
  if (uses_beta == true){
    ss << first_char << "\nconst TFLOAT beta";
    first_char = ',';
  }
  
  ss << ")\n";  
  
}


void BaseGenerator::append_unroll_block_geometry(char x, std::stringstream & ss){

  ss << "\n";
  if (x == 'A') ss << "/* macro tiles define the pattern of C that workgroups (threads with shared local memory) process */\n";
  ss << "#define MACRO_TILE_LENGTH_" << x << " " << hp.at(x).macro_tile_length << "\n";

  if (x == 'A') ss << "/* number of elements in load block : MACRO_TILE_LENGTH_A * UNROLL */\n";
  ss << "#define N_ELEMENTS_IN_" << x << "_UNROLL "<< dp.at(x).n_elements_in_unroll <<"\n";

  if (x == 'A') {
    ss << "/* number of groups covering M : M / MACRO_TILE_LENGTH_A";
    if (dp.main_use_edge_trick == 1){
      ss << " + (PRESHIFT_FINAL_TILE_A != MACRO_TILE_LENGTH_A)";
    }
    ss << " */" << "\n";
  }
  ss << "#define N_GROUPS_" << x << " " <<  dp.at(x).main_n_groups << "\n";

  if (x == 'A') ss << "/* strides parallel to k (unroll) in A. MACRO_STRIDE_ is between unroll tiles, STRIDE_ is within unroll tiles  */\n"; 
  for (std::string orth :  {"PLL", "PERP"}){
    bool pll_k = ("PLL" == orth);
    ss << "#define STRIDE_" << orth << "_K_" << x << " " << dp.get_stride(x, pll_k, false) << "\n";
    ss << "#define MACRO_STRIDE_" << orth << "_K_" << x << " " << dp.get_stride(x, pll_k, true) << "\n";
  }
  
  if (dp.main_use_edge_trick != 0){
    if (x == 'A') ss <<  "/* 1 + (M - 1) % MACRO_TILE_LENGTH_A. somewhere in 1 ... MACRO_TILE_LENGTH_A  */ \n";
    ss << "#define PRESHIFT_FINAL_TILE_" << x << " " << dp.at(x).main_preshift_final_tile << "\n";
  }
}



std::string BaseGenerator::get_time_string(){//  

  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t generation_time = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss <<  
R"(/* ****************************************************************************
* This )" << type << " kernel string was generated on " << std::ctime(&generation_time) << 
R"(**************************************************************************** */)";
  return ss.str();
}

std::string BaseGenerator::get_what_string(){
  return R"(/* ***********************************************
* These parameters define WHAT this kernel does *
* *********************************************** */)";
}

std::string BaseGenerator::get_how_string(){
  return R"(/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/)";
}

std::string BaseGenerator::get_derived_string(){
  return R"(/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */)";
}




}
}


