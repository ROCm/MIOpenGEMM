#include <MIOpenGEMM/copygenerator.hpp>
#include <MIOpenGEMM/error.hpp>

#include <sstream>


namespace MIOpenGEMM{
namespace copygen{



CopyGenerator::CopyGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, const std::string & type_): bylinegen::ByLineGenerator(hp_, gg_, dp_, type_){}


size_t CopyGenerator::get_local_work_size(){
  
 return dp.at(emat_x).cw1_local_work_size;
}

size_t CopyGenerator::get_work_per_thread(){

  return dp.at(emat_x).cw1_work_per_thread;
}

void CopyGenerator::setup_additional() {
    
  if (type.compare("copya") == 0){
    initialise_matrixtype('a');
  }
  
  else if (type.compare("copyb") == 0){
    initialise_matrixtype('b');
  }

  else{
    throw miog_error("Unrecognised type in ByLineGenerator.cpp : " + type + ". should be either copya or copyb \n");
  }

  description_string = R"()";
  inner_work_string = std::string("\n/* the copy */\nw[i] = ") + matrixchar + "[i];";

}


void CopyGenerator::append_derived_definitions_additional(std::stringstream & ss){
  if (matrixchar != 'a' && matrixchar != 'b'){
    throw miog_error(std::string("this is unexpected, call to append_derived_definitions_additional but matrixchar is neither a not b, but rather  ") + matrixchar);
  }
 
  ss << "#define LDW " << dp.get_target_ld(emat_x) << "\n";
  ss << "#define GLOBAL_OFFSET_W " << dp.at(emat_x).cw_global_offset << "\n";
}



KernelString get_copya_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){
 CopyGenerator cg(hp, gg, dp, "copya");
 cg.setup();
 return cg.get_kernelstring();
}

KernelString get_copyb_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){
 CopyGenerator cg(hp, gg, dp, "copyb");
 cg.setup();
 return cg.get_kernelstring();
}




  
}
}
