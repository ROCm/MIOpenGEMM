#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <string>

#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/tinygemmkernelstrings.hpp>

namespace tinygemm{
namespace kerngen{

static const std::string generickernelname = "tg_generated_gemm";

class KernelStringBundle{
  public:
    const std::vector<TinyGemmKernelStrings> v_tgks;
    derivedparams::DerivedParams dp;

    KernelStringBundle(std::vector<TinyGemmKernelStrings> && v_tgks_, const derivedparams::DerivedParams & dp_): v_tgks(v_tgks_), dp(dp_) {}    
};

KernelStringBundle get_kernel_string_bundle(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  

}
}

#endif


//class KernelStringSetStatus{
  //public:
  
    //KernelStringSetStatus(std::string message_):message(message_) {}
    //std::string message;
    
    //bool is_good(){
      //return message == "";
    //}
//};


//std::string && kernel_string_, derivedparams::DerivedParams && dp_, std::string kernel_function_name_


//KernelStringSetStatus set_status;

//std::string kernel_string;
//std::string kernel_function_name;

//std::string && message, 
//set_status(message), 


//kernel_string(kernel_string_), dp(dp_), kernel_function_name(kernel_function_name_) {}
