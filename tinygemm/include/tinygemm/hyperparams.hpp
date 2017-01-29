#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include "tinygemmgeometry.hpp"

namespace tinygemm{
namespace hyperparams{


class HyperParamList{
  
  public:
  
    /* enumerating the hyper parameters */ 
    std::map<std::string, std::string> map_shortkey_to_key;
    std::vector<std::string> keys;
    std::vector<std::string> shortkeys;

    std::string get_key_from_shortkey(const std::string & shortkey);

    HyperParamList();
  
};

extern HyperParamList hpl;
  

class HyperParams{

private: 


public:

  static std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string>> kernel_cache;  
  
  /* the hyper parameters */
  unsigned micro_tile_width;  
  unsigned micro_tile_height;  
  unsigned macro_tile_width; 
  unsigned macro_tile_height;  
  unsigned unroll;  
  
  unsigned pad;  
  unsigned group_allocation;  
  unsigned work_item_load_a_pll_to_unroll;  
  unsigned work_item_load_b_pll_to_unroll;  
  unsigned unroll_pragma;  
  
  unsigned load_to_lds_interwoven;  
  unsigned c_micro_tiles_interwoven; 
  unsigned n_work_items_per_c_elm;  
  unsigned n_target_active_workgroups; 
  unsigned unroll_for_offset;

  std::string get_key_from_shortkey(const std::string & shortkey);
  
  unsigned get_workgroup_size();
  unsigned get_nwitems_h();
  unsigned get_nwitems_w();
  
  HyperParams(const std::map<std::string, unsigned> &);
  
  
  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  
  
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);
  std::vector<HyperParams> get_two_aways(const tinygemm::TinyGemmGeometry & gg);  
  std::map<std::string, unsigned> get_map();
  
  
  std::string get_string() const;


};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);

//HyperParams get_hp(const std::string & hyperstring);

}
}

#endif

