#ifndef BASEGENERATOR_HPP
#define BASEGENERATOR_HPP


#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/derivedparams.hpp>
#include <MIOpenGEMM/kernelstring.hpp>

namespace MIOpenGEMM{
namespace basegen{



class BaseGenerator{


protected:

  const hyperparams::HyperParams & hp;
  const Geometry & gg;
  const derivedparams::DerivedParams & dp;

  std::string type;
  std::string kernelname;
  
  bool uses_a; 
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;

  
  std::string get_time_string();
  
  std::string get_what_string();
  
  std::string get_how_string();
  
  std::string get_derived_string();


public:
  
  /* Does entire setup */
  virtual void setup() = 0;

  virtual KernelString get_kernelstring() = 0;
      
  BaseGenerator(const hyperparams::HyperParams & hp_, const Geometry & gg_, const derivedparams::DerivedParams & dp_, const std::string & type_);

  void append_parameter_list_from_usage(std::stringstream & ss);

  void append_unroll_block_geometry(char x, std::stringstream & ss, bool withcomments, bool with_x_string);
  
  void append_stride_definitions(char x, std::stringstream & ss, unsigned workspace_type, bool withcomments, std::string macro_prefix,  bool append_stride_definitions);


};

}
}

#endif
