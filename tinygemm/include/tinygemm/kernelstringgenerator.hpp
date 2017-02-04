#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <tinygemm/hyperparams.hpp>
#include <string>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>

namespace tinygemm{
namespace kerngen{

class KernelStringSetStatus{
  public:
  
    KernelStringSetStatus(std::string message_):message(message_) {}
    std::string message;
    
    bool is_good(){
      return message == "";
    }
};



class KernelStringBundle{
  public:
    KernelStringSetStatus set_status;
    std::string kernel_string;
    derivedparams::DerivedParams dp;
    std::string kernel_function_name;
    
    KernelStringBundle(std::string && message, std::string && kernel_string_, derivedparams::DerivedParams && dp_, std::string kernel_function_name_):set_status(message), kernel_string(kernel_string_), dp(dp_), kernel_function_name(kernel_function_name_) {}
    
};

KernelStringBundle get_kernel_string_bundle(
  /* hyper parameters */
  const hyperparams::HyperParams & hp,

  std::string kernelname,// = "",


  /* geometry parameters */
  const tinygemm::TinyGemmGeometry & gg

  );


}
}

#endif
