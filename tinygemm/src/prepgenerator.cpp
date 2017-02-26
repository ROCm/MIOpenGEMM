#include <tinygemm/prepgenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>

#include <sstream>

namespace tinygemm{
namespace prepgen{


void PrepGenerator::set_usage_from_matrixchar(){

  uses_alpha = false;  
  if (matrixchar == 'c') {
    uses_a = false;
    uses_b = false;
    uses_c = true;
    uses_workspace = false;
    uses_beta = true;
  }
    
  else{
    uses_c = false;
    uses_workspace = true;
    uses_beta = false;

    if (matrixchar == 'a') {
      uses_a = true;
      uses_b = false;    
    }
    else if (matrixchar =='b') {
      uses_a = false;
      uses_b = true;    
    }
    else{
      throw tinygemm_error("Unrecognised matrixchar in forallgenerator.cpp : " + std::string(1,matrixchar) + std::string(".\n"));
    }
  }
}

void PrepGenerator::append_basic_what_definitions(std::stringstream & ss){
  ss << "#define TFLOAT  "  << dp.t_float << "\n";
  ss << "#define LD" << MATRIXCHAR << " " << gg.get_ld(matrixchar) << "\n";
  ss << "/* less than or equal to LD" << MATRIXCHAR << ", DIM_COAL is size in the contiguous direction (m for c matrix if col contiguous and not transposed) */ \n" << 
"#define DIM_COAL " << gg.get_coal(matrixchar) << "\n" <<
"/* DIM_UNCOAL is the other dimension of the matrix */ \n" << 
"#define DIM_UNCOAL " << gg.get_uncoal(matrixchar) << "\n\n";
}




PrepGenerator::PrepGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string type_): basegen::BaseGenerator(hp_, gg_, dp_, type_){}


}
}

