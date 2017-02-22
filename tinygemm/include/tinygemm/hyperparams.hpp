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
    std::map<std::string, std::string> map_key_to_shortkey;
    std::vector<std::string> keys;
    std::vector<std::string> shortkeys;

    std::string get_key_from_shortkey(const std::string & shortkey);

    HyperParamList();
  
};

extern HyperParamList hpl;
  

class HyperParamsChiral{
  
public:
  unsigned micro_tile_length;
  unsigned macro_tile_length;
  unsigned load_pll_to_unroll;
  unsigned copy_type; //for now 0 : none, 1 : simple copy with ldx padding. To include : 2 : normal form.
  unsigned lds_pad_size;
  
};

class HyperParams{

private: 


public:

  //void bool_check(const std::string & key, unsigned v) const;
  
  //void positive_check(const std::string & key, unsigned v) const;
  
  //void mod_check(const std::string & key1, unsigned v1, const std::string & key2, unsigned v2) const;
  
  void ga_check() const;
  
  void checks() const;
  
  static std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string>> kernel_cache;  
  
  
  HyperParamsChiral aps;
  HyperParamsChiral bps;
  
  const HyperParamsChiral & at(char x) const;
    
  //will go into HyperParamsChiral eventually
  unsigned normal_form;
  //unsigned pad;  

  unsigned unroll;  
  unsigned group_allocation;  
  unsigned unroll_pragma;
  unsigned n_work_items_per_c_elm;  
  unsigned n_target_active_workgroups; 

  //TODO : make theses chiral:
  unsigned load_to_lds_interwoven;
  unsigned c_micro_tiles_interwoven;
  unsigned unroll_for_offset;
 
 

  

  std::string get_key_from_shortkey(const std::string & shortkey);  
  unsigned get_nwitems_h();
  unsigned get_nwitems_w();
  
  HyperParams(const std::map<std::string, unsigned> &);
  
  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);
  std::vector<HyperParams> get_two_aways(const tinygemm::TinyGemmGeometry & gg);  
  std::map<std::string, unsigned> get_params();
  
  
  std::string get_string() const;


  unsigned get_macro_tile_x_length(char x) const;

};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);

//HyperParams get_hp(const std::string & hyperstring);

}
}

#endif

