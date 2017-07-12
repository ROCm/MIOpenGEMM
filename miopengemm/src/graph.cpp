#include <miopengemm/graph.hpp>
#include <miopengemm/macgrid.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <sstream>
#include <algorithm>

namespace MIOpenGEMM{


std::vector<size_t>
get_hy_v(std::string hy_s, bool hy_s_full, Mat::E emat){

  auto p_kv = Mat::mat_to_xchi(emat);
  
  std::vector<size_t> hy_v(p_kv->N, Status::E::UNDEFINED);

  std::vector<std::string> keyvalfrags;
  if (hy_s.compare(""))
  {
    keyvalfrags = stringutil::split(hy_s, "_");
  }

  // MIC, etc
  std::string key;
  // 6, etc
  size_t val;

  auto start = p_kv->name.begin();
  auto end   = p_kv->name.end();
  for (auto& x : keyvalfrags)
  {
    std::tie(key, val) = stringutil::splitnumeric(x);
    if (std::find(start, end, key) == end)
    {
      std::stringstream ss;
      ss << "While processing the constraint string for Sub Graph `" << Mat::M.name[emat] << "', ";
      ss << "the unrecognised key `" + key << "' was not encountered. \n";
      throw miog_error(ss.str());
    }

    size_t keyindex = p_kv->val.at(key);
    
    if (keyindex >= p_kv->N)
    {
      throw miog_error("keyindex exceeds number of sub graph hyper params, internal logic error ");
    }
    
    hy_v[keyindex] = val;
  }

  // A special test in the case that constraints
  // are supposed to be comprehensive
  if (hy_s_full == true)
  {
    for (size_t hpi = 0; hpi < p_kv->N; ++hpi)
    {
      if (hy_v[hpi] == Status::E::UNDEFINED)
      {
        std::stringstream ss;
        ss << "While processing the constraints string of SubG `" << Mat::M.name[emat] << "', ";
        ss << "the parameter `" << p_kv->name[hpi]
           << "' appeared to be unset. Values must all be set as "
           << "hy_s_full is true ";
        throw miog_error(ss.str());
      }
    }
  }

  return hy_v;
}

const std::map<size_t, std::vector<size_t>> g_binary = 
{{Binary::E::NO, {Binary::E::YES}}, {Binary::E::YES, {Binary::E::NO}}};

Graph::Graph(const Geometry & gg, const oclutil::DevInfo & devinfo, const Constraints & cs):
  asubg(gg, cs[Mat::E::A], devinfo), 
  bsubg(gg, cs[Mat::E::B], devinfo),
  csubg(gg, cs[Mat::E::C], devinfo){
    asubg.initialise();
    bsubg.initialise();
    csubg.initialise();  
}


bool Graph::contains(Mat::E emat, size_t hpi, size_t value){
  if (emat >= Mat::E::N){
    throw miog_error("emat not recognised in Graph contains, internal logic error");
  }
  return at(emat).contains(hpi, value);
}


void HyPas::replace_where_defined(const Constraints & constraints){
  for (auto emat :{Mat::E::A, Mat::E::B, Mat::E::C}){
    sus[emat].replace_where_defined(constraints[emat]);
  }
}
  



void HyPas::checks() const{
  for (auto emat :{Mat::E::A, Mat::E::B, Mat::E::C}){
    sus[emat].checks();
  }  
}
  
bool Graph::contains(const HyPas & hp){
  for (auto emat :{Mat::E::A, Mat::E::B, Mat::E::C}){
    if (!at(emat).contains(hp.sus[emat])){
      return false;
    }
  }
  return true;
}


SuGr & Graph::at(size_t emat) {
  switch (emat){
    case Mat::E::A : return asubg;
    case Mat::E::B : return bsubg;
    case Mat::E::C : return csubg;
    default : throw miog_error("unrecogised Mat::E in p_subgs");
  }
}


void SuGr::initialise_range()
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


void SuGr::initialise_start_range()
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

void SuGr::apply_constraints()
{
  throw miog_error("apply constraints to be implemented");
}


void SuGr::initialise()
{
  initialise_edges();  // virtual 
  initialise_range();
  initialise_start_range();
  refine_start_range();  // virtual 
  apply_constraints();
}



void ChiSuGr::initialise_edges()
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
                      {2, {1},
                      }};

  edges[Chi::E::PLU] = {g_binary};
  edges[Chi::E::LIW] = {g_binary};
  edges[Chi::E::MIW] = {g_binary};

  
  edges[Chi::E::WOS] = {{Scratch::E::UNUSED, {Scratch::E::COPY, Scratch::E::NFORM}}, 
    {Scratch::E::COPY, {Scratch::E::UNUSED, Scratch::E::NFORM}}, 
    {Scratch::E::NFORM, {Scratch::E::UNUSED, Scratch::E::COPY}}};
}



void CSuGr::initialise_edges()
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

  edges[NonChi::E::PUN] = {g_binary};
  edges[NonChi::E::UFO] = {g_binary};
}


void ChiSuGr::refine_start_range()
{
  start_range[Chi::E::PAD] = {1, 2};
  start_range[Chi::E::LIW] = {Binary::E::NO};
  start_range[Chi::E::MIW] = {Binary::E::YES};
  start_range[Chi::E::WOS] = {Scratch::E::UNUSED, Scratch::E::COPY, Scratch::E::NFORM};
  
  set_start_mic();
  //refine_start_range_chirality_specific();
}


void CSuGr::refine_start_range()
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


void ChiSuGr::set_start_mic()
{
  
  size_t non_unroll_dimension = ptr_gg->get_non_k_dim(emat);
  
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

SuGr::SuGr(Mat::E e, const Geometry & gg, const Constraint & ct, const oclutil::DevInfo& di):
  emat(e), 
  p_enma(Mat::mat_to_xchi(emat)), 
  ptr_gg(&gg), 
  ptr_constraint(&ct), 
  ptr_devinfo(&di), 
  edges(p_enma->N), 
  range(p_enma->N), 
  start_range(p_enma->N)
  {}

CSuGr::CSuGr(const Geometry & gg, const Constraint & ct, const oclutil::DevInfo& di):     
    SuGr(Mat::E::C, gg, ct, di){}
  
ChiSuGr::ChiSuGr(Mat::E e, const Geometry & gg, const Constraint & ct, const oclutil::DevInfo& di):
  SuGr(e, gg, ct, di){}

ASuGr::ASuGr(const Geometry & gg, const Constraint & constraint, const oclutil::DevInfo& devinfo):
  ChiSuGr(Mat::E::A, gg, constraint, devinfo){}

BSuGr::BSuGr(const Geometry & gg, const Constraint & constraint, const oclutil::DevInfo& devinfo):
  ChiSuGr(Mat::E::B, gg, constraint, devinfo){}


SuHy::SuHy(Mat::E emat): p_enma(Mat::mat_to_xchi(emat)), vs(p_enma->N, Status::E::UNDEFINED){}

SuHy::SuHy(Mat::E emat, const std::string & hyperstring):SuHy(emat){
  vs = get_hy_v(hyperstring, true, emat);
}

HyPas::HyPas(const std::array<std::string, Mat::E::N> & hyperstrings){
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C}){
    sus[emat] = SuHy(emat, hyperstrings[emat]);
  }    
}


bool SuGr::contains(size_t hpi, size_t val){
  if (range.size() >= hpi){
    throw miog_error("in SuGr::contains, range size smaller than hpi, internal logic err");
  }
  bool x = (std::find(range[hpi].begin(), range[hpi].end(), val) == range[hpi].end()) ? false : true;
  return x;
}

bool SuGr::contains(const SuHy & suhy){
  for (size_t hpi = 0; hpi < p_enma->N; ++hpi){    
    if (!contains(hpi, suhy.vs[hpi])){
      return false;
    }
  }
  return true;
}
    
    
}

 
  
  
  
