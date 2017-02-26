#include <tinygemm/copygenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>

#include <sstream>


namespace tinygemm{
namespace copygen{



CopyGenerator::CopyGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_): bylinegen::ByLineGenerator(hp_, gg_, dp_, type_){}


void CopyGenerator::setup_additional() {
    
  if (type.compare("copya") == 0){
    matrixchar = 'a';
    MATRIXCHAR = 'A';
  }
  
  else if (type.compare("copyb") == 0){
    matrixchar = 'b';
    MATRIXCHAR = 'B';
  }

  else{
    throw tinygemm_error("Unrecognised type in ByLineGenerator.cpp : " + type + ". should be either copya or copyb \n");
  }

  description_string = R"()";
  //function_definition = std::string("(__global const TFLOAT * restrict ") + matrixchar + ", const unsigned " + matrixchar + "_offset, __global TFLOAT * w, const unsigned w_offset)";
  inner_work_string = std::string("\n/* the copy */\nw[i] = ") + matrixchar + "[i];";

}


void CopyGenerator::append_derived_definitions_additional(std::stringstream & ss){
  if (matrixchar != 'a' && matrixchar != 'b'){
    throw tinygemm_error(std::string("this is unexpected, call to append_derived_definitions_additional but matrixchar is neither a not b, but rather  ") + matrixchar);
  }
  ss << "#define LDW " << dp.get_target_ld(matrixchar) << "\n";
  ss << "#define GLOBAL_OFFSET_W " << dp.at(matrixchar).cw_global_offset << "\n";
}



KernelString get_copya_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 CopyGenerator cg(hp, gg, dp, "copya");
 cg.setup();
 return cg.get_kernelstring();
}

KernelString get_copyb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 CopyGenerator cg(hp, gg, dp, "copyb");
 cg.setup();
 return cg.get_kernelstring();
}




  
}
}
