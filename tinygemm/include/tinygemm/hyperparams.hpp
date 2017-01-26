#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include "tinygemmgeometry.hpp"

namespace tinygemm{
namespace hyperparams{


static std::vector<std::string> all_hyper_param_names = {"micro_tile_width",  "micro_tile_height",  "macro_tile_width", "macro_tile_height",  "unroll",  "pad",  "group_allocation",  "work_item_load_a_pll_to_unroll",  "work_item_load_b_pll_to_unroll",  "unroll_pragma",  "load_to_lds_interwoven",  "c_micro_tiles_interwoven", "n_work_items_per_c_elm",  "n_target_active_workgroups", "unroll_for_offset"};

class HyperParams{

public:
  std::map<std::string, unsigned> params;
  static std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string>> kernel_cache;
  
  
public:
  unsigned get_workgroup_size();
  unsigned get_nwitems_h();
  unsigned get_nwitems_w();
  void do_checks();
  HyperParams(std::map<std::string, unsigned>);
  HyperParams() = default;
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);
  std::vector<HyperParams> get_two_aways(const tinygemm::TinyGemmGeometry & gg);  
  
  //check that it won't overflow by considering (m,n,tC).
  bool can_be_used_on(const tinygemm::TinyGemmGeometry & gg);  
  std::string get_string() const;


};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);


}
}

#endif

