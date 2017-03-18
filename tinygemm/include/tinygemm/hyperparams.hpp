#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include "tinygemmgeometry.hpp"

namespace tinygemm{
namespace hyperparams{

/* enumerating parameters */ 
class ParamList{

  private:
    /* private, as get_key_from_shortkey is the preferred access route due to it's checks. Logic : minimise direct map accesses */ 
    std::map<std::string, std::string> map_key_to_shortkey;
    std::map<std::string, std::string> map_shortkey_to_key;
    
  public:
    std::vector<std::string> keys;
    std::vector<std::string> shortkeys;    
    
    std::string get_key_from_shortkey(const std::string & shortkey) const;
    std::string get_shortkey_from_key(const std::string & shortkey) const;

    ParamList(std::map<std::string, std::string> bla_to_bla);
  
};

extern ParamList nonchiral_pl;
extern ParamList chiral_pl;



class HyperParam{
  public:
      
    unsigned importance;
    unsigned val;
    
    HyperParam();

    virtual std::vector<unsigned> get_one_away() const = 0;
    virtual std::string get_key() const = 0;
    virtual std::string get_shortkey() const = 0;
    virtual bool is_valid() const = 0;
    
    std::string get_string(){      
      return get_shortkey() + std::to_string(val);
    };
    
};



/* with a fixed and enumerable transition graph / matrix */
class SimpleHyperParam : public HyperParam{
  private:


  public:
    virtual std::vector<unsigned> get_one_away() const override final { return one_aways[val] };
    virtual std::string get_key() const = 0;
    virtual std::string get_shortkey() const = 0;
    virtual bool is_valid() const override final {return true; } // TODO should check in range
    SimpleHyperParam(std::vector<unsigned, std::vector<unsigned> > one_aways_):one_aways(one_aways_){} //TODO should set range
};

class hp_MIC : public SimpleHyperParam{

public:

  /* the transition graph / matrix */
  const static std::vector<unsigned, std::vector<unsigned> > one_aways;
  
  /* vertices in transition graph */
  const static std::vector<unsigned> range;
    
  virtual std::string get_key() const override final { return "micro_tile_length"; }
  virtual std::string get_shortkey() const override final { return "MIC"; };
  hp_MIC(): SimpleHyperParam{
    {1, {2,3,4} },
    {2, {1,3,4} },
    {3, {1,2,4,5} },
    {4, {1,2,3,5,6} },
    {5, {2,3,4,6,7} },
    {6, {3,4,5,7,8} },
    {7, {4,5,6,8} },
    {8, {4,6,7} }
  };
}

class hp_U : public SimpleHyperParam{  
public:
  virtual std::string get_key() const override final { return "unroll"; }
  virtual std::string get_shortkey() const override final { return "U"; };
  hp_U(): SimpleHyperParam{
    {4, {8} },
    {8, {4,16} },
    {16, {8,24,32} },
    {24, {16, 32} },
    {32, {16} }
  }
};


class hp_MAC : public SimpleHyperParam {
  
  
  /* 
   * 0 : 4,  8
   * 1 : 8,  4
   * 2 : 8,  8
   * 3 : 16, 8
   * 4 : 8,  16
   * 5 : 16, 16
   * 
   * */
  
public:
  /* in derived ? */
  void set_macro_tile_lengths(unsigned & maca, unsigned & macb){
    if      (val == 0) {maca = 4; macb = 8;}
    else if (val == 1) {maca = 8; macb = 4;}
    else if (val == 2) {maca = 8; macb = 8;}
    else if (val == 3) {maca = 8; macb = 16;}
    else if (val == 4) {maca = 16; macb = 8;}
    else if (val == 5) {maca = 16; macb = 16;}
  }
    

  virtual std::string get_string() const override final {
    return get_shortkey() + std::to_string(val);
  }
     
  std::string get_noncryptic_string() {
    unsigned maca; 
    unsigned macb;
    set_macro_tile_lengths(maca, macb);
    std::stringstream ss;
    ss << get_shortkey() << "A" << maca << "B" << macb;
    return  ss.str();
  }

  virtual std::string get_key() const override final { return "work_item_grid"; }
  virtual std::string get_shortkey() const override final { return "MAC"; };
  hp_MAC(): SimpleHyperParam{
    //TODO : this is not maintainable. use enums? A8B16 --> 0 or something ?
    {0, {2} },
    {1, {2} },
    {2, {0,1,3,4,5}},
    {3, {2,5}},
    {4, {2,5}}
  }
  
};




class ChiralHyperParams{
  
public:
  hp_MIC micro_tile_length;
  hp_MAC macro_tile_length(this);
  hp_PLU load_pll_to_unroll;
  hp_WOS workspace_type;
  hp_PAD lds_pad_size;
  hp_LIW load_to_lds_interwoven;
  hp_MIW c_micro_tiles_interwoven;
  
  void cw_check() const;
  void checks() const;
  std::string get_string() const;
  
  ChiralHyperParams();
  
};



class HyperParams{

private: 

public:
  
  void ga_check() const;
  
  void checks() const;
  
  static std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string>> kernel_cache;  
  
  
  ChiralHyperParams aps;
  ChiralHyperParams bps;
  
  const ChiralHyperParams & at(char x) const;
  ChiralHyperParams & at(char x);  

    
  hp_U unroll;  
  hp_GA group_allocation;  
  hp_PU unroll_pragma;
  hp_ICE n_work_items_per_c_elm;  
  hp_NAW n_target_active_workgroups; 
  hp_UFO unroll_for_offset;


  std::string get_key_from_shortkey(const std::string & shortkey);  
  
  unsigned get_nwitems_h();
  unsigned get_nwitems_w();
  
  HyperParams(const std::map<char, std::map<std::string, unsigned> > & );

  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);
  std::vector<HyperParams> get_two_aways(const tinygemm::TinyGemmGeometry & gg);  
  std::map<char, std::map<std::string, unsigned > > get_params();
  
  void check_map_keys(const std::map<char, std::map<std::string, unsigned> > & params);
  std::string get_string() const;
  
  unsigned get_macro_tile_x_length(char x) const;

};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);

}
}

#endif

