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
  
  //unsigned micro_tile_width = 3, 
  //unsigned micro_tile_height = 2, 
  //unsigned macro_tile_width = 48, 
  //unsigned macro_tile_height = 32, 
  //unsigned unroll = 16, 
  //unsigned pad = 1, 
  //unsigned group_allocation = 2, 
  //unsigned work_item_load_a_pll_to_unroll = 0, 
  //unsigned work_item_load_b_pll_to_unroll = 1, 
  //unsigned unroll_pragma = 1, 
  //unsigned load_to_lds_interwoven = 0, 
  //unsigned c_micro_tiles_interwoven = 1, 
  //unsigned n_work_items_per_c_elm = 3,
  //unsigned unroll_for_offset = 1,
  //unsigned n_target_active_workgroups = 64);
 
 
  

   



}
}

#endif
