#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <tinygemm/hyperparams.hpp>
#include <string>

namespace tinygemm{
namespace kerngen{

class KernelStringSetStatus{
  public:
  
    KernelStringSetStatus(std::string message):message(message) {}
    
    std::string message;
    bool is_good(){
      return message == "";
    }
  
};

KernelStringSetStatus set_kernel_string(
  /* hyper parameters */
  const hyperparams::HyperParams & hp,
  
  std::string & kernel_string,
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
