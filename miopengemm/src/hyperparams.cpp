/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/randomutil.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/macgrid.hpp>

namespace MIOpenGEMM
{


// TODO : this is clogging up the namespace, fix.
RandomUtil radu;

template <typename T>
std::map<T, size_t> get_vals(size_t nVals, const std::vector<T>& keys, const std::string& hash)
{
  std::map<T, size_t> vals;
  for (size_t val = 0; val < nVals; ++val)
  {
    if (keys[val] == T())
    {
      throw miog_error("It appears as though one of the elements of " + hash +
                       " has not been added to keys, unitialisation error");
    }
    vals[keys[val]] = val;
  }
  return vals;
}


std::vector<std::string> get_sub_constraints(std::string constraints_string)
{
  std::vector<std::string> sub_constraints(Mat::E::N, "");
  auto                     megafrags = stringutil::split(constraints_string, "__");
  for (auto& megafrag : megafrags)
  {
    if (Mat::M.val.count(megafrag[0]) == 0)
    {
      std::stringstream ss;
      ss << "\nWhile reading hyperstring in get-params-from-string,\n";
      ss << "the leading char should be A,B or C, not `" << megafrag[0] << "'.\n";
      throw miog_error(ss.str());
    }
    if (megafrag.size() < 3)
    {
      std::stringstream ss;
      ss << "sub constraint " << megafrag << " is too short, something is wrong. \n";
      throw miog_error(ss.str());
    }
    sub_constraints[Mat::M.val.at(megafrag[0])] = megafrag.substr(2);
  }
  return sub_constraints;
}

Graph::Graph(const Geometry&                     gg,
             const oclutil::DevInfo& devinfo,
             std::string                         constraints_string,
             bool                                full_cs)
  : ptr_gg(&gg)
{

  constraints_string_in                    = constraints_string;
  std::vector<std::string> sub_constraints = get_sub_constraints(constraints_string);

  asubg = ASubG(gg, sub_constraints[Mat::E::A], full_cs, &devinfo);
  asubg.initialise();

  bsubg = BSubG(gg, sub_constraints[Mat::E::B], full_cs, &devinfo);
  bsubg.initialise();

  csubg = CSubG(gg, sub_constraints[Mat::E::C], full_cs, &devinfo);
  csubg.initialise();

  p_subgs[Mat::E::A] = &asubg;
  p_subgs[Mat::E::B] = &bsubg;
  p_subgs[Mat::E::C] = &csubg;


  // TODO liberate these : they should not belong to one graph!
  coupled_parameters.push_back({{Mat::E::A, Chi::E::MIC}, {Mat::E::B, Chi::E::MIC}});
  coupled_parameters.push_back({{Mat::E::C, NonChi::E::UFO}, {Mat::E::C, NonChi::E::PUN}});
  coupled_parameters.push_back({{Mat::E::C, NonChi::E::UNR}, {Mat::E::C, NonChi::E::ICE}});
}





void Graph::force_start_node(std::string constraints_string)
{
  std::vector<std::string> sub_constraints = get_sub_constraints(constraints_string);

  auto a_vals =
    get_constraints(sub_constraints[Mat::E::A], true, asubg.ptr_keys_vals, asubg.get_char());
  auto b_vals =
    get_constraints(sub_constraints[Mat::E::B], true, bsubg.ptr_keys_vals, bsubg.get_char());
  auto c_vals =
    get_constraints(sub_constraints[Mat::E::C], true, csubg.ptr_keys_vals, csubg.get_char());

  asubg.force_start_node(a_vals);
  bsubg.force_start_node(b_vals);
  csubg.force_start_node(c_vals);
}

std::vector<std::vector<size_t>> get_all_constraints(std::string constraints_string)
{

  std::vector<std::string>           sub_constraints = get_sub_constraints(constraints_string);
  std::vector<std::vector<size_t>> all_constraints(Mat::E::N);

  all_constraints[Mat::E::A] =
    get_constraints(sub_constraints[Mat::E::A], false, &Chi::M, 'A');
  all_constraints[Mat::E::B] =
    get_constraints(sub_constraints[Mat::E::B], false, &Chi::M, 'B');
  all_constraints[Mat::E::C] =
    get_constraints(sub_constraints[Mat::E::C], false, &NonChi::M, 'C');

  return all_constraints;
}

std::vector<size_t>
get_constraints(std::string subg_cs, bool subg_csfull, const EnumMapper<std::string> * p_kv, char subg_hash)
{

  std::vector<size_t> constraints(p_kv->N, Status::E::UNDEFINED);

  std::vector<std::string> keyvalfrags;
  if (subg_cs.compare(""))
  {
    keyvalfrags = stringutil::split(subg_cs, "_");
  }

  // MIC, etc
  std::string key;
  // 6, etc
  size_t val;
  for (auto& x : keyvalfrags)
  {
    std::tie(key, val) = stringutil::splitnumeric(x);
    auto start = p_kv->name.begin();
    auto end   = p_kv->name.end();
    if (std::find(start, end, key) == end)
    {
      std::stringstream ss;
      ss << "While processing the constraint string for SubG `" << subg_hash << "', ";
      ss << "the key `" + key << "' was not recognised. In set_constraints(). \n";
      throw miog_error(ss.str());
    }

    size_t keyindex = p_kv->val.at(key);
    if (keyindex < constraints.size())
    {
      constraints[keyindex] = val;
    }

    else
    {
      throw miog_error("in get constrains, strange out of bounds error, come "
                       "and investigate");
    }
  }

  // A special test in the case that constraints
  // are supposed to be comprehensive
  if (subg_csfull == true)
  {
    for (size_t hpi = 0; hpi < p_kv->N; ++hpi)
    {
      if (constraints[hpi] == Status::E::UNDEFINED)
      {
        std::stringstream ss;
        ss << "While processing the constraints string of SubG `" << subg_hash << "', ";
        ss << "the parameter `" << p_kv->name[hpi]
           << "' appeared to be unset. The constraints must all be set "
              "(subg_csfull is true) \n";
        throw miog_error(ss.str());
      }
    }
  }

  return constraints;
}

void SubG::set_constraints()
{
  constraints = get_constraints(subg_cs, subg_csfull, ptr_keys_vals, get_char());
}

const std::map<size_t, std::vector<size_t>> graph_binary = {{0, {1}}, {1, {0}}};

void SubG::initialise_range_from_preconstraint_edges()
{
  range.resize(edges.size());
  for (size_t hpi = 0; hpi < edges.size(); ++hpi)
  {
    for (auto& x : edges[hpi])
    {
      range[hpi].push_back(x.first);
    }
  }
}

void SubG::initialise_start_range_from_range()
{
  start_range.resize(range.size());
  for (size_t hpi = 0; hpi < range.size(); ++hpi)
  {
    for (auto& x : range[hpi])
    {
      start_range[hpi].push_back(x);
    }
  }
}

void SubG::force_start_node(std::vector<size_t> start_node)
{

  if (start_node.size() != range.size())
  {
    std::stringstream ss;

    ss << "in force_start_node, and start_node.size() (=" << start_node.size()
       << ") differs from range.size() << (" << range.size() << ")";
    throw miog_error(ss.str());
  }

  for (size_t hpi = 0; hpi < range.size(); ++hpi)
  {
    start_range[hpi] = {start_node.at(hpi)};
  }
}

SubG::SubG(size_t                            nHPs_,
           const Geometry&                     gg,
           std::string                         cs,
           bool                                csfull,
           const oclutil::DevInfo* ptr_devinfo_)
  : nHPs(nHPs_),
    ptr_gg(&gg),
    edges(nHPs_),
    start_range(nHPs_),
    subg_cs(cs),
    subg_csfull(csfull),
    ptr_devinfo(ptr_devinfo_)
{
}

void ChiralSubG::initialise_maps() { ptr_keys_vals = &Chi::M; }

void CSubG::initialise_maps() { ptr_keys_vals = &NonChi::M; }

void SubG::initialise()
{
  initialise_maps();
  set_constraints();
  set_preconstraint_edges();
  initialise_range_from_preconstraint_edges();
  initialise_start_range_from_range();
  manual_override_start_range();
  apply_constraints();
  confirm_start_is_subset();
}

std::string SubG::get_edges_string(size_t hpi)
{
  std::stringstream ss;
  ss << "Edges : \n";
  for (auto& key_vec : edges[hpi])
  {
    ss << key_vec.first << " :  ";
    for (auto v : key_vec.second)
    {
      ss << v << " ";
    }
    ss << '\n';
  }
  return ss.str();
}

std::string get_generic_range_string(std::string                  opener,
                                     const std::vector<size_t>& generic_range_hpi)
{
  std::stringstream ss;
  ss << opener << " : \n";
  for (auto& x : generic_range_hpi)
  {
    ss << x << " ";
  }
  ss << '\n';
  return ss.str();
}

std::string SubG::get_range_string(size_t hpi)
{
  return get_generic_range_string("Range", range[hpi]);
}

std::string SubG::get_start_range_string(size_t hpi)
{
  return get_generic_range_string("Start Range", start_range[hpi]);
}

std::string SubG::get_string(size_t hpi)
{
  std::stringstream ss;
  ss << get_edges_string(hpi);
  ss << get_range_string(hpi);
  ss << get_start_range_string(hpi);

  ss << "Start Range : \n";
  for (auto& x : start_range[hpi])
  {
    ss << x << " ";
  }
  ss << '\n';
  return ss.str();
}

void SubG::confirm_start_is_subset()
{

  for (size_t hpi = 0; hpi < nHPs; ++hpi)
  {
    if (start_range[hpi].size() == 0)
    {
      std::stringstream ss;
      ss << "no valid value to start from in " << ptr_keys_vals->name[hpi];
      throw miog_error(ss.str());
    }

    for (auto& x : start_range[hpi])
    {
      if (std::count(range[hpi].begin(), range[hpi].end(), x) == 0)
      {
        std::stringstream ss;
        ss << "It seems like the start_range element `" << x << "' is not in the range of "
           << ptr_keys_vals->name[hpi] << ".";
        ss << "The full setup of " << ptr_keys_vals->name[hpi] << " is\n ";
        ss << get_string(hpi);
        throw miog_error(ss.str());
      }
    }
  }
}

CSubG::CSubG(const Geometry&                     gg,
             std::string                         cs,
             bool                                csfull,
             const oclutil::DevInfo* ptr_devinfo_)
  : SubG(NonChi::E::N, gg, cs, csfull, ptr_devinfo_)
{
}

ChiralSubG::ChiralSubG(const Geometry&                     gg,
                       std::string                         cs,
                       bool                                csfull,
                       const oclutil::DevInfo* ptr_devinfo_)
  : SubG(Chi::E::N, gg, cs, csfull, ptr_devinfo_)
{
}

void ChiralSubG::set_chirality_specific_start_range_base(size_t non_unroll_dimension)
{
  std::vector<size_t> basemic = {8, 6};
  if (non_unroll_dimension < 256)
  {
    basemic.push_back(5);
    basemic.push_back(4);
  }

  if (non_unroll_dimension < 128)
  {
    basemic.push_back(3);
    basemic.push_back(2);
  }

  if (non_unroll_dimension < 64)
  {
    basemic.push_back(1);
  }

  start_range[Chi::E::MIC] = {};
  for (auto& x : basemic)
  {
    if (x <= non_unroll_dimension)
    {
      start_range[Chi::E::MIC].push_back(x);
    }
  }
}

void ASubG::set_chirality_specific_start_range()
{
  set_chirality_specific_start_range_base(ptr_gg->m);
}

void BSubG::set_chirality_specific_start_range()
{
  set_chirality_specific_start_range_base(ptr_gg->n);
}

void ChiralSubG::manual_override_start_range()
{
  start_range[Chi::E::PAD] = {1, 2};
  start_range[Chi::E::LIW] = {Binary::E::NO};
  start_range[Chi::E::MIW] = {Binary::E::YES};
  start_range[Chi::E::WOS] = {Scratch::E::UNUSED, Scratch::E::COPY, Scratch::E::NFORM};
  set_chirality_specific_start_range();
}

void SubG::apply_constraints()
{
  for (size_t hpi = 0; hpi < nHPs; ++hpi)
  {
    if (constraints.at(hpi) != Status::E::UNDEFINED)
    {

      if (ptr_devinfo->device_name != "unknown_default_constructed")
      {
        if (std::find(range[hpi].begin(), range[hpi].end(), constraints.at(hpi)) ==
            range[hpi].end())
        {
          std::stringstream errm;
          errm << "the constraint on " << ptr_keys_vals->name[hpi] << " of " << constraints.at(hpi)
               << " is not in the pre-constraint range:  \n"
               << get_range_string(hpi);
          errm << "this is not currently allowed";
          throw miog_error(errm.str());
        }
      }

      edges[hpi]       = {{constraints.at(hpi), {}}};
      range[hpi]       = {constraints.at(hpi)};
      start_range[hpi] = {constraints.at(hpi)};
    }
  }
}

void ChiralSubG::set_preconstraint_edges()
{

  edges[Chi::E::MIC] = {{1, {2, 3}},
                      {2, {1, 3, 4}},
                      {3, {1, 2, 4}},
                      {4, {2, 3, 5, 6}},
                      {5, {2, 4, 6}},
                      {6, {4, 5, 8}},
                      {8, {4, 6}}};

  edges[Chi::E::PAD] = {{0, {1}},
                      {1, {0, 2}},
                      {
                        2, {1},
                      }};

  edges[Chi::E::PLU] = {graph_binary};
  edges[Chi::E::LIW] = {graph_binary};
  edges[Chi::E::MIW] = {graph_binary};

  
  edges[Chi::E::WOS] = {{Scratch::E::UNUSED, {Scratch::E::COPY, Scratch::E::NFORM}}, 
    {Scratch::E::COPY, {Scratch::E::UNUSED, Scratch::E::NFORM}}, 
    {Scratch::E::NFORM, {Scratch::E::UNUSED, Scratch::E::COPY}}};
}


void CSubG::manual_override_start_range()
{

  start_range[NonChi::E::UNR] = {8, 16};
  start_range[NonChi::E::ICE] = {1};
  start_range[NonChi::E::UFO] = {Binary::E::NO};

  if ((ptr_gg->m) > 200 && (ptr_gg->n) > 200)
  {

    if (ptr_devinfo->wg_atom_size == 32)
    {
      start_range[NonChi::E::SKW] = {macgrid::skew0, macgrid::skew0 + 1};
    }

    else
    {
      start_range[NonChi::E::SKW] = {macgrid::skew0};
    }
  }
}

void CSubG::set_preconstraint_edges()
{

  edges[NonChi::E::UNR] = {{8, {16}}, {16, {8, 32}}, {32, {16, 64}}, {64, {16, 32}}};
  edges[NonChi::E::NAW] = {{64, {16}}, {16, {64}}};
  edges[NonChi::E::GAL] = 
  {{GroupAllocation::E::BYROW,  {GroupAllocation::E::BYCOL, GroupAllocation::E::SUCOL}},
  {GroupAllocation::E::BYCOL,   {GroupAllocation::E::BYROW, GroupAllocation::E::SUCOL}},
  {GroupAllocation::E::SUCOL,   {GroupAllocation::E::BYROW, GroupAllocation::E::BYCOL}}};

  // MAC and SKW

  if (ptr_devinfo->device_name == "unknown_default_constructed")
  {
  }

  else if (ptr_devinfo->wg_atom_size != 64 && ptr_devinfo->wg_atom_size != 32)
  {
    std::stringstream ss;
    ss << "(device_name : " << ptr_devinfo->device_name << ")  "
       << "Setting up the edge search graph in set_preconstraint_edges, and it "
          "seems like the "
          "atomic wg size is neither 32 or 64. Is this correct ?? If so, "
          "consider changing here or "
          "raise an issue";
    throw miog_error(ss.str());
  }

  // very small / thin matrices
  else if (ptr_gg->m * ptr_gg->n < 32 * 32 || ptr_gg->m < 16 || ptr_gg->n < 16)
  {
    edges[NonChi::E::MAC] = {
      {1, {4, 16}}, {4, {1, 16, 64}}, {16, {4, 64}}, {64, {16, 256}}, {256, {64}},
    };

    edges[NonChi::E::SKW] = {

      {7, {8}},
      {8, {7, 9}},
      {9, {8, 10}},
      {10, {9, 11}},
      {11, {10, 12}},
      {12, {11, 13}},
      {13, {12}},

    };
  }

  else if (ptr_devinfo->wg_atom_size == 64)
  {
    edges[NonChi::E::MAC] = {{64, {256}}, {256, {64}}};
    edges[NonChi::E::SKW] = {{9, {10}}, {10, {9, 11}}, {11, {10}}};
  }

  else if (ptr_devinfo->wg_atom_size == 32)
  {
    edges[NonChi::E::MAC] = {{32, {64, 256}}, {64, {32, 128, 256}}, {128, {64, 256}}, {256, {64}}};
    edges[NonChi::E::SKW] = {{9, {10}}, {10, {9, 11}}, {11, {10, 12}}, {12, {10, 11}}};
  }

  else
  {
    throw miog_error("wg_atom_size is neither 32 or 64, how can this be? I "
                     "thought we'd already "
                     "checked this. (Logic error)");
  }

  edges[NonChi::E::ICE] = {{1, {2}},
                      {2, {1, 3, 4}},
                      {3, {1, 2, 4, 6}},
                      {4, {1, 3, 5, 7}},
                      {5, {1, 2, 4, 6, 8}},
                      {6, {1, 3, 5, 7, 9}},
                      {7, {4, 6, 8, 10}},
                      {8, {1, 5, 7, 9, 11}},
                      {9, {6, 8, 10, 12}},
                      {10, {1, 7, 9, 11, 13}},
                      {11, {8, 10, 12, 14}},
                      {12, {1, 9, 11, 13, 14}},
                      {13, {10, 12, 14}},
                      {14, {1, 11, 13}}};

  edges[NonChi::E::PUN] = {graph_binary};
  edges[NonChi::E::UFO] = {graph_binary};
}

void HyperParams::checks() const
{
  for (size_t gi = 0; gi < Mat::E::N; ++gi)
  {
    if (gi > v_xhps.size())
    {
      throw miog_error("strange error : gi > v_xhps.size()");
    }

    const XHPs& x     = v_xhps[gi];
    SubG&       sub_g = *(p_graph->p_subgs[gi]);
    for (size_t hpi = 0; hpi < sub_g.nHPs; ++hpi)
    {
      if (hpi >= sub_g.range.size())
      {
        std::stringstream errm;
        errm << "strange error : hpi >= graph.range.size()\n";
        errm << "specifically, " << hpi << " >= " << sub_g.range.size();
        throw miog_error(errm.str());
      }

      auto start = sub_g.range[hpi].begin();
      auto end   = sub_g.range[hpi].end();

      if (x.vs[hpi] == Status::E::UNDEFINED || (std::find(start, end, x.vs[hpi]) == end))
      {

        std::stringstream errm;
        errm << "\nIn HyperParams::checks(). It appears as though `" << x.vs[hpi]
             << "' is not a valid value for " << sub_g.ptr_keys_vals->name[hpi] << ".\n";
        errm << "the relevant graph looks like this: \n" << sub_g.get_string(hpi);
        throw miog_error(errm.str());
      }
    }
  }
}


// go through the params, and where it is not nHP::UNDEFINED,
// use its value to replace this
void HyperParams::replace_where_source_defined(const std::vector<std::vector<size_t>>& params)
{
  for (size_t mi = 0; mi < Mat::E::N; ++mi)
  {
    for (size_t hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi)
    {
      if (params[mi][hpi] != Status::E::UNDEFINED)
      {
        v_xhps[mi].vs[hpi] = params[mi][hpi];
      }
    }
  }
}

HyperParams::HyperParams(const Graph& graph) : p_graph(&graph)
{
  for (size_t mi = 0; mi < Mat::E::N; ++mi)
  {
    v_xhps.emplace_back(XHPs(p_graph->p_subgs[mi]->nHPs));
    for (size_t hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi)
    {
      auto&    a_range   = p_graph->p_subgs[mi]->start_range[hpi];
      size_t index     = radu.get_from_range(a_range.size());
      v_xhps[mi].vs[hpi] = a_range[index];
    }
  }
  checks();
}

bool HyperParams::operator==(const HyperParams& hpr) { return get_string() == hpr.get_string(); }

std::string HyperParams::get_part_string(char X) const
{
  size_t          mi = Mat::M.val.at(X);
  std::stringstream ss;
  ss << X;
  for (size_t hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi)
  {
    ss << "_" << p_graph->p_subgs[mi]->ptr_keys_vals->name[hpi] << v_xhps[mi].vs[hpi];
  }
  return ss.str();
}

std::string HyperParams::get_string() const
{
  std::stringstream ss;
  ss << get_part_string('A') << "__" << get_part_string('B') << "__" << get_part_string('C');
  return ss.str();
}

std::vector<HyperParams> HyperParams::get_one_aways()
{

  std::vector<HyperParams> one_aways;

  // by changing just one hyper-parameter
  // TODO : should changing an inactive parameter be allowed? Like, when GAL != 3, should GAW?
  for (size_t mi = 0; mi < Mat::E::N; ++mi)
  {
    for (size_t hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi)
    {
      size_t value = v_xhps[mi].vs[hpi];
      for (auto& newval : p_graph->p_subgs[mi]->edges[hpi].at(value))
      {
        HyperParams hp(*this);
        hp.v_xhps[mi].vs[hpi] = newval;
        one_aways.push_back(hp);
      }
    }
  }

  // by changing MAC and one or both MICs,
  // so as to semi-preserve the overall
  // shape of the macro tile
  size_t curr_mac = v_xhps[Mat::E::C].vs[NonChi::E::MAC];
  for (auto& newmac : p_graph->p_subgs[Mat::E::C]->edges[NonChi::E::MAC].at(curr_mac))
  {

    // ratios of new to current tile grid sizes
    auto curr_grid_size_tuple = macgrid::get_grid(curr_mac, v_xhps[Mat::E::C].vs[NonChi::E::SKW]);
    auto curr_grid_size       = std::get<2>(curr_grid_size_tuple);

    auto new_grid_size_tuple = macgrid::get_grid(newmac, v_xhps[Mat::E::C].vs[NonChi::E::SKW]);
    if (std::get<0>(new_grid_size_tuple) == false)
    {
      continue;
    }
    auto new_grid_size = std::get<2>(new_grid_size_tuple);

    double delta_na = static_cast<double>(new_grid_size[Mat::E::A]) /
                      static_cast<double>(curr_grid_size[Mat::E::A]);
    double delta_nb = static_cast<double>(new_grid_size[Mat::E::B]) /
                      static_cast<double>(curr_grid_size[Mat::E::B]);

    // mica scaled so that the macro tile
    // remains ~ the same in the a dimension

    size_t curr_mica = v_xhps[Mat::E::A].vs[Chi::E::MIC];
    size_t new_mica  = static_cast<size_t>(static_cast<double>(curr_mica) / delta_na);

    // micb scaled so that the macro tile remains the same in the b dimension
    size_t curr_micb = v_xhps[Mat::E::B].vs[Chi::E::MIC];
    size_t new_micb  = static_cast<size_t>(static_cast<double>(curr_micb) / delta_nb);

    // if the new micro tile (a) is different and valid, add it
    if (new_mica != curr_mica && in_graph(Mat::E::A, Chi::E::MIC, new_mica))
    {
      HyperParams hp(*this);
      hp.v_xhps[Mat::E::C].vs[NonChi::E::MAC] = newmac;
      hp.v_xhps[Mat::E::A].vs[Chi::E::MIC] = new_mica;
      one_aways.push_back(hp);
    }

    if (new_micb != curr_micb && in_graph(Mat::E::B, Chi::E::MIC, new_micb))
    {
      HyperParams hp(*this);
      hp.v_xhps[Mat::E::C].vs[NonChi::E::MAC] = newmac;
      hp.v_xhps[Mat::E::B].vs[Chi::E::MIC] = new_micb;
      one_aways.push_back(hp);

      if (new_mica != curr_mica && in_graph(Mat::E::A, Chi::E::MIC, new_mica))
      {
        HyperParams hp2(hp);
        hp2.v_xhps[Mat::E::A].vs[Chi::E::MIC] = new_mica;
        one_aways.push_back(hp2);
      }
    }
  }

  size_t n_uncoupled = one_aways.size();

  // by changing two hyper-parameters
  for (auto& couple_p : p_graph->coupled_parameters)
  {

    auto first       = std::get<0>(couple_p);
    auto first_m     = std::get<0>(first);
    auto first_p     = std::get<1>(first);
    auto first_value = v_xhps[first_m].vs[first_p];

    auto second       = std::get<1>(couple_p);
    auto second_m     = std::get<0>(second);
    auto second_p     = std::get<1>(second);
    auto second_value = v_xhps[second_m].vs[second_p];

    for (auto& new_first_val : p_graph->p_subgs[first_m]->edges[first_p].at(first_value))
    {
      for (auto& new_second_val : p_graph->p_subgs[second_m]->edges[second_p].at(second_value))
      {

        // only if one increases and one decreases
        if ((new_second_val > second_value) != (new_first_val > first_value))
        {
          HyperParams hp(*this);
          hp.v_xhps[first_m].vs[first_p]   = new_first_val;
          hp.v_xhps[second_m].vs[second_p] = new_second_val;
          one_aways.push_back(hp);
        }
      }
    }
  }

  size_t n_total = one_aways.size();

  // shuffle the true one aways
  radu.shuffle(0, n_uncoupled, one_aways);

  // shuffle the two aways (coupled)
  radu.shuffle(n_uncoupled, n_total, one_aways);

  // shuffle the custom kernels if there are any (old)

  return one_aways;
}

bool HyperParams::in_graph(size_t mi, size_t hpi, size_t value)
{
  return std::count(p_graph->p_subgs[mi]->range[hpi].begin(),
                    p_graph->p_subgs[mi]->range[hpi].end(),
                    value) != 0;
}

std::tuple<bool, std::string> HyperParams::in_graph()
{
  std::string in_graph_string("in graph");
  // filtering out if violates the constraint string
  bool constraints_satisfied = true;
  for (size_t mi = 0; mi < Mat::E::N; ++mi)
  {
    for (size_t hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi)
    {
      if (in_graph(mi, hpi, v_xhps[mi].vs[hpi]) == false)
      {

        std::stringstream sstr;
        sstr << "hyper param : " << p_graph->p_subgs[mi]->ptr_keys_vals->name[hpi] << ", and value "
             << v_xhps[mi].vs[hpi] << ".";
        in_graph_string       = sstr.str();
        constraints_satisfied = false;
        break;
      }
    }
  }

  return std::make_tuple(constraints_satisfied, in_graph_string);
}


}
