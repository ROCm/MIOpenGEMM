/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_HYPERKERNELPARAMS_HPP
#define GUARD_MIOPENGEMM_HYPERKERNELPARAMS_HPP

#include <array>
#include <array>
#include <functional>
#include <map>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/openclutil.hpp>

namespace MIOpenGEMM
{



namespace nsMAC
{

std::tuple<bool, std::string, std::array<size_t, 2>> get_mac_grid(size_t mac, size_t skew);
}



namespace hyperparams
{

std::vector<size_t>
get_constraints(std::string subg_cs, bool subg_csfull, const EnumMapper<std::string> * p_kv, char subg_hash);


class SubG
{

  public:
  SubG(size_t                            nHPs,
       const Geometry&                     gg,
       std::string                         cs,
       bool                                csfull,
       const openclutil::OpenCLDeviceInfo* ptr_devinfo);

  SubG() = default;

  void apply_constraints();

  size_t        nHPs;
  const Geometry* ptr_gg;

  const EnumMapper<std::string>* ptr_keys_vals;

  // all the possible edges from all the possible hyper parameter
  // example : edges[Chi::E::MIC] is a map; edges[Chi::E::MIC][1] --> {2,3,4}
  std::vector<std::map<size_t, std::vector<size_t>>> edges;

  // all the possible values of a hyper parameter
  // example : range[Chi::E::MIC] --> {1,2,3,4,5,6,7,8}
  std::vector<std::vector<size_t>> range;

  // a subset of range, the possible values returned on a request for a random value
  // example : start_range[Chi::E::MIC] --> {2,8}. It can depend on geometry (from initialisation)
  std::vector<std::vector<size_t>> start_range;

  std::string subg_cs;
  bool        subg_csfull;

  const openclutil::OpenCLDeviceInfo* ptr_devinfo;

  std::vector<size_t> constraints;

  void         initialise();
  void         set_constraints();
  void         initialise_range_from_preconstraint_edges();
  void         initialise_start_range_from_range();
  void         confirm_start_is_subset();
  virtual void initialise_maps()         = 0;
  virtual void set_preconstraint_edges() = 0;
  // used if start range should be a strict subset of range
  virtual void manual_override_start_range() = 0;
  virtual char get_char()                    = 0;

  std::string get_string(size_t hpi);
  std::string get_edges_string(size_t hpi);
  std::string get_range_string(size_t hpi);
  std::string get_start_range_string(size_t hpi);

  void force_start_node(std::vector<size_t>);
};

class CSubG : public SubG
{
  public:
  CSubG() = default;
  CSubG(const Geometry&                     gg,
        std::string                         cs,
        bool                                csfull,
        const openclutil::OpenCLDeviceInfo* ptr_devinfo);
  virtual void initialise_maps() override final;
  virtual void set_preconstraint_edges() override final;
  virtual void manual_override_start_range() override final;
  virtual char get_char() override final { return 'C'; }
};

class ChiralSubG : public SubG
{
  public:
  ChiralSubG() = default;
  ChiralSubG(const Geometry&                     gg,
             std::string                         cs,
             bool                                csfull,
             const openclutil::OpenCLDeviceInfo* ptr_devinfo);
  virtual void initialise_maps() override final;
  virtual void set_preconstraint_edges() override final;
  virtual void manual_override_start_range() override final;
  void set_chirality_specific_start_range_base(size_t non_unroll_dimension);
  virtual void set_chirality_specific_start_range() = 0;
};

class ASubG : public ChiralSubG
{
  public:
  ASubG() = default;
  ASubG(const Geometry&                     gg,
        std::string                         cs,
        bool                                csfull,
        const openclutil::OpenCLDeviceInfo* ptr_devinfo_)
    : ChiralSubG(gg, cs, csfull, ptr_devinfo_)
  {
  }
  virtual void set_chirality_specific_start_range() override final;
  virtual char get_char() override final { return 'A'; }
};

class BSubG : public ChiralSubG
{
  public:
  BSubG() = default;
  BSubG(const Geometry&                     gg,
        std::string                         cs,
        bool                                csfull,
        const openclutil::OpenCLDeviceInfo* ptr_devinfo_)
    : ChiralSubG(gg, cs, csfull, ptr_devinfo_)
  {
  }
  virtual void set_chirality_specific_start_range() override final;
  virtual char get_char() override final { return 'B'; }
};

class Graph
{

  private:
  ASubG asubg;
  BSubG bsubg;
  CSubG csubg;

  public:
  const Geometry* ptr_gg;

  /* TODO if a Graph is copied this causes undefined behaviour.
   * can Graphs be copied? Make design clearer */
  std::vector<SubG*> p_subgs;

  std::string constraints_string_in;

  void force_start_node(std::string);

  std::vector<std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>>
    coupled_parameters;

  Graph(const Geometry&                     gg,
        const openclutil::OpenCLDeviceInfo& devinfo,
        std::string                         constraint_string,
        bool                                full_constraints_expected);
};

class XHPs
{

  public:
  std::vector<size_t> vs;
  XHPs(size_t nHPs) { vs = std::vector<size_t>(nHPs, Status::E::UNDEFINED); }
};

class HyperParams
{

  private:
  const Graph*      p_graph;
  std::vector<XHPs> v_xhps;

  public:
  void replace_undefined_randomly();
  void replace(const std::vector<std::vector<size_t>>& partial_params);
  void replace_where_source_defined(const std::vector<std::vector<size_t>>& params);
  bool in_graph(size_t mi, size_t hpi, size_t value);
  std::tuple<bool, std::string> in_graph();

  const XHPs& at(Mat::E subgtype) const { return v_xhps[subgtype]; }
  XHPs& at(Mat::E subgtype) { return v_xhps[subgtype]; }

  HyperParams(const Graph& graph);

  bool operator==(const HyperParams& hpr);
  std::vector<HyperParams> get_one_aways();
  std::string get_part_string(char X) const;
  std::string get_string() const;
  void        checks() const;
};

std::vector<std::vector<size_t>> get_all_constraints(std::string constraints_string);
}
}

#endif
