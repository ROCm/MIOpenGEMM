#ifndef BETACGENERATOR_HPP
#define BETACGENERATOR_HPP

#include <tinygemm/bylinegenerator.hpp>

namespace tinygemm{
namespace betacgen{


class BetacGenerator : public bylinegen::ByLineGenerator {

public:

BetacGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_);


//: bylinegen::ByLineGenerator(hp_, gg_, dp_, "betac"){}


  virtual void setup_additional() final override;
  
  
  ////{
    ////matrixchar = 'c';

    ////description_string = R"(
/////* ****************************************************
////* It is used to perform the beta*C step in GEMM, 
////* where recall GEMM has C <- alpha*A*B + beta*C
////* It is not quite an axpy, as when ldc is not minimal, 
////* C is not contiguous memory  
////****************************************************** */ )";
    ////function_definition = "(__global TFLOAT * x, const unsigned x_offset, TFLOAT beta)";
    ////inner_work_string = "\n/* the beta scaling */\nx[i] *= beta;";


  ////}

    
  
};



KernelString get_betac_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

//{


 //BetacGenerator bcg(hp, gg, dp);
 //bcg.setup();
 //return bcg.get_kernelstring();
   
//}

  
}
}

#endif


