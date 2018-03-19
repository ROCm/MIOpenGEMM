/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DERIVEDPARAMS_HPP
#define GUARD_MIOPENGEMM_DERIVEDPARAMS_HPP

#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>

namespace MIOpenGEMM
{

const size_t uninitialised_size_t = std::numeric_limits<size_t>::max();

class Derivabilty
{
  public:
  bool        is_derivable;
  std::string msg;
  Derivabilty(const HyPas&, const Geometry&);
};

bool is_dvble(const HyPas&, const Geometry&);

class ChiralDerivedParams
{
  public:
  size_t macro_tile_length                    = uninitialised_size_t;
  size_t n_elements_in_unroll                 = uninitialised_size_t;
  size_t main_n_elements_to_load_per_workitem = uninitialised_size_t;
  size_t main_n_elements_in_padded_unroll     = uninitialised_size_t;
  size_t main_micro_tile_pll_unroll           = uninitialised_size_t;
  size_t main_micro_tile_perp_unroll          = uninitialised_size_t;
  size_t main_n_micro_tiles_pll_unroll        = uninitialised_size_t;
  size_t main_macro_tile_length_and_pad       = uninitialised_size_t;
  size_t main_n_micro_in_macro                = uninitialised_size_t;
  size_t preshift_final_tile                  = uninitialised_size_t;

  // how many macro_lengths to cover m (a) or n (b)
  size_t n_groups = uninitialised_size_t;

  // used when loading LDS -> registers, depends on MIW
  size_t main_c_interweave_stride;

  // copy to workspace specific parameters
  size_t cw_global_offset = uninitialised_size_t;
  size_t cw_n_elements    = uninitialised_size_t;

  // copy to workspace, type 1, specific parameters
  size_t cw1_smallest_possible_ldx = uninitialised_size_t;
  size_t cw1_target_ldx            = uninitialised_size_t;
  size_t cw1_local_work_size       = uninitialised_size_t;
  size_t cw1_work_per_thread       = uninitialised_size_t;

  // copy to workspace, type 2, specific parameters
  size_t cw2_local_work_size           = uninitialised_size_t;
  size_t cw2_load_pll_to_unroll        = uninitialised_size_t;  // always perp
  size_t cw2_micro_tile_pll_unroll     = uninitialised_size_t;
  size_t cw2_micro_tile_perp_unroll    = uninitialised_size_t;
  size_t cw2_n_micro_tiles_pll_unroll  = uninitialised_size_t;
  size_t cw2_n_micro_tiles_perp_unroll = uninitialised_size_t;

  size_t cw2_n_elements_perp_unroll          = uninitialised_size_t;
  size_t cw2_n_elements_to_load_per_workitem = uninitialised_size_t;

  std::string get_string();
};

// all derived parameters
class DerivedParams
{

  private:
  const HyPas*    ptr_hp;
  const Geometry* ptr_gg;

  ChiralDerivedParams adps{};
  ChiralDerivedParams bdps{};

  void reset_ga3_params();

  void reset_cw_params(Mat::E emat_x);

  public:
  // initiate all parameters, throwing an error if there is an incompatibility
  DerivedParams(const HyPas& hp, const Geometry& gg);

  DerivedParams(const HyPas& hp, const Geometry& gg, std::string s);

  DerivedParams()                     = delete;
  DerivedParams(const DerivedParams&) = delete;
  DerivedParams(DerivedParams&&)      = default;

  ChiralDerivedParams& at(Mat::E emat_x) { return emat_x == Mat::E::A ? adps : bdps; }
  const ChiralDerivedParams& at(Mat::E emat_x) const { return emat_x == Mat::E::A ? adps : bdps; }

  // does the minimum setting to confirm compatibitily.
  std::tuple<bool, std::string> set_fragile();

  size_t main_macro_tile_area = uninitialised_size_t;
  size_t main_micro_tile_area = uninitialised_size_t;

  size_t main_n_work_items_per_workgroup = uninitialised_size_t;
  size_t main_n_work_groups              = uninitialised_size_t;
  size_t main_global_work_size           = uninitialised_size_t;

  size_t main_split_on_k              = uninitialised_size_t;
  size_t main_does_beta_c_inc         = uninitialised_size_t;
  size_t main_use_edge_trick          = uninitialised_size_t;
  size_t main_final_fractional_unroll = uninitialised_size_t;

  // specific to scaling kernel, betac
  size_t betac_local_work_size = uninitialised_size_t;
  size_t betac_work_per_thread = uninitialised_size_t;

  size_t cw2_n_macro_tiles_pll_unroll = uninitialised_size_t;

  // the int type for atomics
  std::string infa;
  // the function to use for atomic ints
  std::string fati;
  // one of __K_NORMAL_FORM__   __K__  and  k_plus_offset
  std::string effective_k_varies_string;
  // as their names suggest
  std::string k_effective_mod_G_UNROLL;
  std::string k_effective_div_G_UNROLL;
  std::string k_effective_div_UNROLL;

  // pragma unroll string : #pragma unroll\n or ""
  std::string pragma_unroll_string;
  //* currently one of "float" and "double", set from float_size
  std::string t_float;

  // GA 3 specific derived parameters
  size_t ga3_super_column_width      = uninitialised_size_t;
  size_t ga3_last_super_column_width = uninitialised_size_t;

  // Total required workspace
  size_t required_workspace = uninitialised_size_t;

  size_t get_target_ld(Mat::E emat_x) const;

  size_t get_n_elements_in_x_unroll(char x);

  size_t get_stride(Mat::E emat_x, bool pll_k, bool is_macro, size_t workspace_type) const;

  size_t get_stride_cw0(Mat::E emat_x, bool pll_k) const;

  size_t get_stride_cw1(Mat::E emat_x, bool pll_k) const;

  size_t get_stride_cw2(Mat::E emat_x, bool pll_k, bool is_macro) const;

  std::array<std::string, Mem::E::N> tints;
  std::string tintk;
  std::string tshort;

  void set_should_be_hyperparams();

  std::string get_string();
};
}

#endif
