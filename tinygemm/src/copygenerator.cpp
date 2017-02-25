#include <tinygemm/copygenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>

#include <sstream>


namespace tinygemm{
namespace copygen{



CopyGenerator::CopyGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_): bylinegen::ByLineGenerator(hp_, gg_, dp_, type_){}


void CopyGenerator::setup_additional() {
    
  if (type.compare("copya") == 0){
    matrixchar = 'a';
  }
  
  else if (type.compare("copyb") == 0){
    matrixchar = 'b';
  }

  else{
    throw tinygemm_error("Unrecognised type in ByLineGenerator.cpp : " + type + ". should be either copya or copyb \n");
  }

  description_string = R"()";
  function_definition = "(__global const TFLOAT * restrict x, const unsigned x_offset, __global TFLOAT * y, const unsigned y_offset)";
  inner_work_string = "\n/* the copy */\ny[i] = x[i];";

}


void CopyGenerator::append_derived_definitions_additional(std::stringstream & ss){
  if (matrixchar != 'a' && matrixchar == 'b'){
    throw tinygemm_error("this is unexpected, call to append_derived_definitions_additional but matrixchar is neither a not b");
  }
  ss << "#define GLOBAL_OFFSET_Y " << dp.at(matrixchar).cw_global_offset << "\n";
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
