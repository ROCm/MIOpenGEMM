#ifndef BASEGENERATOR_HPP
#define BASEGENERATOR_HPP


#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/kernelstring.hpp>

namespace tinygemm{
namespace basegen{



class BaseGenerator{


protected:

  const tinygemm::hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  const tinygemm::derivedparams::DerivedParams & dp;

  std::string type;
  std::string kernelname;
  //std::string function_definition;
  
  bool uses_a; 
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  //std::string get_generic_kernelname(const std::string & type);
  
  std::string get_time_string();//const std::string & type); 
  
  std::string get_what_string();
  
  std::string get_how_string();
  
  std::string get_derived_string();


public:
  
  /* Does entire setup */
  virtual void setup() = 0;

  virtual KernelString get_kernelstring() = 0;
      
  BaseGenerator(const tinygemm::hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, const std::string & type_);

  void append_parameter_list_from_usage(std::stringstream & ss);


  void append_unroll_block_geometry(char x, std::stringstream & ss);

};

}
}

#endif
