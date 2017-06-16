#include <MIOpenGEMM/prepgenerator.hpp>
#include <MIOpenGEMM/error.hpp>

#include <sstream>

namespace MIOpenGEMM{
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
      throw miog_error("Unrecognised matrixchar in forallgenerator.cpp : " + std::string(1,matrixchar) + std::string(".\n"));
    }
  }
}

void PrepGenerator::append_basic_what_definitions(std::stringstream & ss){
  ss << "#define TFLOAT  "  << dp.t_float << "\n";
  ss << "#define LD" << MATRIXCHAR << " " << gg.ldX.at(emat_x) << "\n";
  ss << "/* less than or equal to LD" << MATRIXCHAR << ", DIM_COAL is size in the contiguous direction (m for c matrix if col contiguous and not transposed) */ \n" << 
"#define DIM_COAL " << gg.get_coal(emat_x) << "\n" <<
"/* DIM_UNCOAL is the other dimension of the matrix */ \n" << 
"#define DIM_UNCOAL " << gg.get_uncoal(emat_x) << "\n\n";
}




PrepGenerator::PrepGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, std::string type_): basegen::BaseGenerator(hp_, gg_, dp_, type_){}


}
}

