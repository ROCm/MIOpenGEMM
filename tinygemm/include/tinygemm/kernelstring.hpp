#ifndef TINYGEMMKERNELSTRINGS_HPP
#define TINYGEMMKERNELSTRINGS_HPP

#include <string>


#include <iostream>
namespace tinygemm{

class KernelString{
public:
  /*type : betac_alphab, betac_workspace, etc.*/
  std::string type;  
  std::string kernstr;
  std::string fname;
  
  size_t global_work_size;
  size_t local_work_size;

  KernelString(const std::string & type_, std::string && kernstr_, const std::string & fname_ , size_t global_work_size_,  size_t local_work_size_): type(type_), kernstr(kernstr_), fname(fname_), global_work_size(global_work_size_), local_work_size(local_work_size_) {
  
  }

  KernelString() = default;

  
};

  
}

#endif
