#ifndef TINYGEMMKERNELSTRINGS_HPP
#define TINYGEMMKERNELSTRINGS_HPP

#include <string>

namespace tinygemm{

class KernelString{
public:
  /*type : betac_alphab, betac_workspace, etc.*/
  std::string type;  
  std::string kernstr;
  std::string fname;
  KernelString(std::string && type_, std::string && kernstr_, const std::string & fname_  ):type(type_), kernstr(kernstr_), fname(fname_) {}

  KernelString() = default;

};

  
}

#endif
