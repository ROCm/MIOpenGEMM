#include <tinygemm/kernelstring.hpp>

#include <tinygemm/tinygemmerror.hpp>
#include <iostream>


namespace tinygemm{
  


bool KernelType::uses(char c) const{
  if (c == 'a'){
    return uses_a;
  }
  else if (c == 'b'){
    return uses_b;
  }
  else if (c == 'c'){
    return uses_c;
  }
  else if (c == 'w'){
    return uses_workspace;
  }
  else{
    throw tinygemm_error(std::string("unrecognised char in uses in KernelType, ") + c);
  }
}

KernelType::KernelType(bool uses_a_, bool uses_b_, bool uses_c_, bool uses_workspace_, bool uses_alpha_, bool uses_beta_):
uses_a(uses_a_), uses_b(uses_b_), uses_c(uses_c_), uses_workspace(uses_workspace_), uses_alpha(uses_alpha_), uses_beta(uses_beta_)
{
  for (auto & x : {'a', 'b', 'c', 'w'}){
    if (uses(x)){
      full += x;
    }
  }
  if (uses_alpha){
    full += "_alpha";
  }
  
  if (uses_beta){
    full += "_beta";
  }
  
  /* TODO: are we sure that the main kernel will always use alpha ? */
  if (uses_alpha){
    basic = "main";
  }
  
  else if (uses_beta && uses('c')){
    basic  = "betac";
  }
  
  else if (uses('a') && uses('w')){
    basic = "copya";
  }
  
  else if (uses('b') && uses('w')){
    basic = "copyb";
  }
  
  else{
    throw tinygemm_error("determining `basic' string of KernelType, not sure what this kernel does. Its full string is " + full);
  }
    
}




}
