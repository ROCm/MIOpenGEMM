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


  //static const HyperParamList hpl;
  void add_parameter_in_constructor(unsigned & up, const std::string & sp, const std::map<std::string, unsigned> & params);  
  
  /* an alternative way to access the hyper parameters, via a key to a map */
  std::map<std::string, unsigned * > pval_from_key;




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
  unsigned get_val_from_shortkey(const std::string & shortkey) const;
  
  
  unsigned get_workgroup_size();
  unsigned get_nwitems_h();
  unsigned get_nwitems_w();
  //void do_checks();
  
  HyperParams(const std::map<std::string, unsigned> &);
  //HyperParams() = default;
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);
  std::vector<HyperParams> get_two_aways(const tinygemm::TinyGemmGeometry & gg);  
  std::map<std::string, unsigned> get_map();
  
  //check that it won't overflow by considering (m,n,tC).
  bool can_be_used_on(const tinygemm::TinyGemmGeometry & gg);  
  std::string get_string() const;

  void add_hyperparam(const std::string & hyperstring, std::vector<HyperParams> & one_aways);

};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);

HyperParams get_hp(const std::string & hyperstring);

}
}

#endif

