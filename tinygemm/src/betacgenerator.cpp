#include <MIOpenGEMM/betacgenerator.hpp>

#include <sstream>
#include <iostream>
namespace MIOpenGEMM{
namespace betacgen{



BetacGenerator::BetacGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_): bylinegen::ByLineGenerator(hp_, gg_, dp_, "betac"){
}

size_t BetacGenerator::get_local_work_size(){
  return dp.betac_local_work_size;
}

size_t BetacGenerator::get_work_per_thread(){
  return dp.betac_work_per_thread;
}


void BetacGenerator::setup_additional(){
  initialise_matrixtype('c');

  description_string = R"(
/* ****************************************************
* It is used to perform the beta*C step in GEMM, 
* where recall GEMM has C <- alpha*A*B + beta*C
* It is not quite an axpy, as when ldc is not minimal, 
* C is not contiguous memory  
****************************************************** */ )";
  inner_work_string = "\n/* the beta scaling */\nc[i] *= beta;";

}


void BetacGenerator::append_derived_definitions_additional(std::stringstream & ss) {    
  ss << " ";
}
  



KernelString get_betac_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){
 BetacGenerator bcg(hp, gg, dp);
 bcg.setup();
 return bcg.get_kernelstring();
}

  
}
}

