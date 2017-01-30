#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <tinygemm/hyperparams.hpp>
#include <string>
#include <tinygemm/derivedparams.hpp>

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
  
  //std::string & kernel_string,
  
  std::string kernelname = "",
  /* geometry parameters */
  unsigned float_size = 32,
  unsigned a_transposed = 1,
  unsigned b_transposed = 0,
  unsigned c_transposed = 0,
  unsigned is_col_major = 1
  /* to add m,n,k, lda, ldb, ldc */
  );


}
}

#endif
