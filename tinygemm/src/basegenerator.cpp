#include <tinygemm/basegenerator.hpp>
#include <tinygemm/generatorutil.hpp>

#include <sstream>

namespace tinygemm{
namespace basegen{

      
BaseGenerator::BaseGenerator(const tinygemm::hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_): hp(hp_), gg(gg_), dp(dp_), type(type_), kernelname(genutil::get_generic_kernelname(type)) {}
  

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
    ss << first_char << "\n__global const TFLOAT * restrict workspace, \n" << "const unsigned workspace_offset";
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


}
}


