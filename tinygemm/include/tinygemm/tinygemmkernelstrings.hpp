#ifndef TINYGEMMKERNELSTRINGS_HPP
#define TINYGEMMKERNELSTRINGS_HPP

#include <string>

namespace tinygemm{

class TinyGemmKernelStrings{
public:
  /*type : betac_alphab, betac_workspace, etc.*/
  std::string type;  
  std::string kernstr;
  std::string fname;
  TinyGemmKernelStrings(std::string && type_, std::string && kernstr_, const std::string & fname_  ):type(type_), kernstr(kernstr_), fname(fname_) {}

  TinyGemmKernelStrings() = default;

};

  
}

#endif
