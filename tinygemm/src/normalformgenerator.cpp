#include <string>
#include <sstream>

#include <tinygemm/normalformgenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>

//TODO : inheritence, for forallgenerator, alphagenerator and normalformgenerator. 

namespace tinygemm{
namespace nformgen{

const size_t n_work_items_per_group = 256;

class NormalFormGenerator : public prepgen::PrepGenerator{

private:

  /* these should all go to derived params. That way, is_deriveable can be called first to ensure tileability. */
  unsigned work_item_load_pll_to_unroll = 0;
  unsigned micro_tile_pll_unroll;
  unsigned micro_tile_perp_unroll;
  unsigned n_macro_tiles_pll_unroll;  
  unsigned n_macro_tiles;
  
  
  
  unsigned stride_pll_unroll;
  unsigned stride_perp_unroll;
  unsigned n_micro_pll_unroll;
  unsigned n_micro_perp_unroll;
  unsigned global_offset_y;


public:
  NormalFormGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string type_): prepgen::PrepGenerator(hp_, gg_, dp_, type_){}

  void setup() override final{
    

    if (type.compare("nforma") == 0){
      matrixchar = 'a';
    }
    
    else if (type.compare("nformb") == 0){
      matrixchar = 'b';
    }
  
    else{
      throw tinygemm_error("Unrecognised type in normalformgenerator.cpp : " + type + ". should be either nforma or nformb \n");
    }
    
    set_usage_from_matrixchar();

  }
  
  size_t get_local_work_size() override final{
    /* should be made into a hyper param */
    return 64;
  }

  size_t get_n_work_groups() override final{
    return 101010101;
  }











KernelString get_kernelstring(){
  std::stringstream ss;
  
  append_unroll_block_geometry(matrixchar, ss);
  
  //get id_pll_unroll and id_perp_unroll  
  ss << "not done yet";
  return {{uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta}, ss.str(), kernelname, get_global_work_size(), get_local_work_size()};
}


};




KernelString get_nforma_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 NormalFormGenerator nfg(hp, gg, dp, "nforma");
 nfg.setup();
 return nfg.get_kernelstring();
}

KernelString get_nformb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 NormalFormGenerator nfg(hp, gg, dp, "nformb");
 nfg.setup();
 return nfg.get_kernelstring();
}


}
}



