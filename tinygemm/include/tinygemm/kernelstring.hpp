#ifndef TINYGEMMKERNELSTRINGS_HPP
#define TINYGEMMKERNELSTRINGS_HPP









#include <string>


namespace tinygemm{




class KernelType{

public:

/* summary of uses_a, uses_b, uses_c etc */
std::string full;

/* one of copya, copyb, betac, main */
std::string basic;

bool uses_a;
bool uses_b;
bool uses_c;
bool uses_workspace;
bool uses_alpha;
bool uses_beta;  

bool uses(char c) const;

KernelType(bool uses_a_, bool uses_b_, bool uses_c_, bool uses_workspace_, bool uses_alpha_, bool uses_beta_);

KernelType() = default;

};


class KernelString{
public:
  /*type : betac_alphab, betac_workspace, etc.*/
  // std::string type;  
  KernelType type;
  std::string kernstr;
  std::string fname;
  
  size_t global_work_size;
  size_t local_work_size;

  KernelString(const KernelType & type_, std::string && kernstr_, const std::string & fname_ , size_t global_work_size_,  size_t local_work_size_): type(type_), kernstr(kernstr_), fname(fname_), global_work_size(global_work_size_), local_work_size(local_work_size_) {
  
  }

  KernelString() = default;

  
};

  
}

#endif
