#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <limits>

#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/mapkeycheck.hpp>
#include <tinygemm/randomutil.hpp>

namespace tinygemm{

/* macro tile shape */
/* at skew = skew0, (64, 256) are a:b = 1:1 and (32, 128) have a:b = 2:1 */
unsigned skew0 = 10;  
  
namespace nsMAC{

std::array<unsigned, 2> get_mac_grid(unsigned mac, unsigned skew){

  double dbl_lg2_mac = std::log2 (static_cast<double> (mac));
  unsigned lg2_mac = static_cast<unsigned> (dbl_lg2_mac);
  

  double na = std::exp2(lg2_mac/2 + lg2_mac%2);
  double nb = static_cast<double> (mac) / na;

  /* at skew = skew0 + 1, (64, 256) are a:b = 1:4 and (32, 128) have a:b = 1:2 */
  for (unsigned i = skew0; i < skew; ++i){
    na /= 2.;
    nb *= 2.;
  }   
  
  /* at skew0 = skew + 1, (64, 256) are a:b = 4:1 and (32, 128) have a:b = 8:1 */
  for (unsigned i = skew; i < skew0; ++i){
    na *= 2.;
    nb /= 2.;
  }

  unsigned u_na = static_cast<unsigned> (na);
  unsigned u_nb = static_cast<unsigned> (nb);
  
  if (std::abs (na*nb - static_cast<double> (u_na*u_nb)) > 1e-7){
    throw tinygemm_error("error in getting mac sizes, casting non-ints ");
  }
  
  
  std::array<unsigned, 2> mac_grid;
  
  if (nsHP::matA >= 2 || nsHP::matB >= 2){
    throw tinygemm_error("the std::array returned in get_mac_grid is too small");
  }
  
  mac_grid[nsHP::matA] = u_na;
  mac_grid[nsHP::matB] = u_nb;
      
  return mac_grid;
}


}

namespace hyperparams{

RandomUtil radu;

template <typename T>
std::map<T, unsigned> getVals(unsigned nVals, const std::vector<T> & keys, const std::string & hash){
  std::map<T, unsigned> vals;    
  for (unsigned val = 0; val < nVals; ++val){
    if (keys[val] == T()){
      throw tinygemm_error("It appears as though one of the elements of " + hash +  " has not been added to keys, unitialisation error");
    }
    vals[keys[val]] = val;
    
  }
  return vals;
}


Graph::Graph(const tinygemm::TinyGemmGeometry & gg, const openclutil::OpenCLDeviceInfo & devinfo, std::string constraints_string, bool full_cs): ptr_gg(&gg) {

  graphchar.resize(nsHP::nMats);
  graphchar[nsHP::matA] = 'A';
  graphchar[nsHP::matB] = 'B';
  graphchar[nsHP::matC] = 'C';
  
  graphind = getVals(nsHP::nMats, graphchar, "graphind");

  std::vector<std::string> sub_constraints(nsHP::nMats, "");
  auto megafrags = stringutil::split(constraints_string, "__");

  
  for (auto & megafrag : megafrags){
    if (graphind.count(megafrag[0]) == 0){
      std::stringstream ss;
      ss << "\nWhile reading hyperstring in get-params-from-string,\n";
      ss << "the leading char should be A,B or C, not `" << megafrag[0] << "'.\n";
      throw tinygemm_error(ss.str());
    }
    if (megafrag.size() < 3){
      std::stringstream ss;
      ss << "sub constraint " << megafrag << " is too short, something is wrong. \n";
      throw tinygemm_error(ss.str());
    }
    sub_constraints[graphind[megafrag[0]]] = megafrag.substr(2);
  }
  
  asubg = ASubG(gg, sub_constraints[nsHP::matA], full_cs);
  asubg.initialise();
  bsubg = BSubG(gg, sub_constraints[nsHP::matB], full_cs);
  bsubg.initialise();
  csubg = CSubG(gg, sub_constraints[nsHP::matC], full_cs);
  csubg.initialise();

  p_subgs.resize(nsHP::nMats);
  p_subgs[nsHP::matA] = &asubg;
  p_subgs[nsHP::matB] = &bsubg;
  p_subgs[nsHP::matC] = &csubg;

  coupled_parameters.push_back( { {nsHP::matA, nsHP::MIC}, {nsHP::matB, nsHP::MIC} } );
  coupled_parameters.push_back( { {nsHP::matC, nsHP::UFO}, {nsHP::matC, nsHP::PUN} } );
  coupled_parameters.push_back( { {nsHP::matC, nsHP::UNR}, {nsHP::matC, nsHP::ICE} } );


}

void SubG::set_constraints(){
  
  constraints = std::vector<unsigned> (nHPs, nsHP::undefined);
  
  std::vector<std::string> keyvalfrags;
  if (subg_cs.compare("")){
    keyvalfrags = stringutil::split(subg_cs, "_");
  }
  
  
  /* MIC, etc. */
  std::string key; 
  /* 6, etc */
  unsigned val;  
  for (auto & x : keyvalfrags){
    std::tie(key, val) = stringutil::splitnumeric(x);
    auto start = Keys.begin();
    auto end  = Keys.end();
    if(std::find(start, end, key) == end) {
      std::stringstream ss;
      ss << "While processing the constraint string for SubG `" << get_char() << "', ";
      ss << "the key `" + key << "' was not recognised. In set_constraints(). \n";
      throw tinygemm_error(ss.str());
    }

    unsigned keyindex = Vals.at(key);
    if (keyindex < constraints.size()){
      constraints[keyindex] = val;
    }

    else{
      throw tinygemm_error("in get constrains, strange out of bounds error, come and investigate");
    }
  }


  /* A special test in the case that constraints are supposed to be comprehensive */
  if (subg_csfull == true)  {
    for (unsigned hpi = 0; hpi < nHPs; ++hpi){
      if (constraints[hpi] == nsHP::undefined){
        std::stringstream ss;
        ss << "While processing the constraints string of SubG `" << get_char() << "', ";
        ss << "the parameter `" << Keys[hpi] << "' appeared to be unset. The constraints must all be set (subg_csfull is true) \n";
        throw tinygemm_error(ss.str()); 
      }
    }
  }  
}



void ChiralSubG::initialise_maps(){
  Keys[nsHP::MIC] = "MIC";
  Keys[nsHP::PAD] = "PAD";
  Keys[nsHP::PLU] = "PLU";
  Keys[nsHP::LIW] = "LIW";
  Keys[nsHP::MIW] = "MIW";
  Keys[nsHP::WOS] = "WOS";
  Vals = getVals(nsHP::nChiralHPs, Keys, "ChiralKeys");
}

void CSubG::initialise_maps(){
  Keys[nsHP::UNR] = "UNR";
  Keys[nsHP::GAL] = "GAL";
  Keys[nsHP::PUN] = "PUN";
  Keys[nsHP::ICE] = "ICE";
  Keys[nsHP::NAW] = "NAW";
  Keys[nsHP::UFO] = "UFO"; 
  Keys[nsHP::MAC] = "MAC";
  Keys[nsHP::SKW] = "SKW"; 
  Vals = getVals(nsHP::nNonChiralHPs, Keys, "NonChiralKeys");
}



const std::map<unsigned, std::vector<unsigned> > graph_binary = 
{   {0, {1}},
    {1, {0}}    };

void SubG::initialise_range_from_preconstraint_edges(){
  range.resize(edges.size());
  for (unsigned hpi = 0; hpi < edges.size(); ++hpi){
    for (auto & x : edges[hpi]){
      range[hpi].push_back(x.first);
    }
  }
}


SubG::SubG(unsigned nHPs_, const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull): nHPs(nHPs_), ptr_gg(&gg), Keys(nHPs_), edges (nHPs_), start_range (nHPs_), subg_cs(cs), subg_csfull(csfull) {
  
  
}

void SubG::initialise(){
  initialise_maps();
  set_constraints();
  set_preconstraint_edges();
  initialise_range_from_preconstraint_edges();
  set_start_range();
  apply_constraints();
  confirm_start_is_subset();
}


std::string SubG::get_edges_string(unsigned hpi){
  std::stringstream ss;
  ss << "Edges : \n";
  for (auto & key_vec : edges[hpi]){
    ss << key_vec.first << " :  ";
    for (auto v : key_vec.second){
      ss << v << " ";
    }
    ss << "\n";
  }
  return ss.str();
}


std::string get_generic_range_string(std::string opener, const std::vector<unsigned> & generic_range_hpi){
  std::stringstream ss;
  ss << opener << " : \n";
  for (auto & x : generic_range_hpi){
    ss << x << " ";
  }
  ss << "\n";
  return ss.str();
}

std::string SubG::get_range_string(unsigned hpi){
  return get_generic_range_string("Range", range[hpi]);
}

std::string SubG::get_start_range_string(unsigned hpi){
  return get_generic_range_string("Start Range", start_range[hpi]);
}


std::string SubG::get_string(unsigned hpi){
  std::stringstream ss;
  ss << get_edges_string(hpi);
  ss << get_range_string(hpi);
  ss << get_start_range_string(hpi);  

  ss << "Start Range : \n";
  for (auto & x : start_range[hpi]){
    ss << x << " ";
  }
  ss << "\n"; 
  return ss.str();
}

void SubG::confirm_start_is_subset(){
  
  for (unsigned hpi = 0; hpi < nHPs; ++hpi){    
    if (start_range[hpi].size() == 0){
      std::stringstream ss;
      ss << "no valid value to start from in " << Keys[hpi];
      throw tinygemm_error(ss.str());
    }
    
    for (auto & x : start_range[hpi]){
      if (std::count(range[hpi].begin(), range[hpi].end(), x) == 0){
        std::stringstream ss;
        ss << "It seems like the start_range element `" << x << "' is not in the range of " << Keys[hpi] << ".";
        ss << "The full setup of " << Keys[hpi] << " is\n ";
        ss << get_string(hpi);
        throw tinygemm_error(ss.str());
      }
    }
  }
}


CSubG::CSubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull) : SubG(nsHP::nNonChiralHPs, gg, cs, csfull){}

ChiralSubG::ChiralSubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull) : SubG(nsHP::nChiralHPs, gg, cs, csfull){}


void ChiralSubG::set_chirality_specific_start_range_base(unsigned non_unroll_dimension){
  start_range[nsHP::MIC] = {8,6};
  if (non_unroll_dimension  < 256){
    start_range[nsHP::MIC].push_back(5);
    start_range[nsHP::MIC].push_back(4);
  }
  
  if (non_unroll_dimension < 128){
    start_range[nsHP::MIC].push_back(3);
    start_range[nsHP::MIC].push_back(2);
  }
  
  if (non_unroll_dimension < 64){
    start_range[nsHP::MIC].push_back(1);
  }
}

void ASubG::set_chirality_specific_start_range(){
  set_chirality_specific_start_range_base(ptr_gg->m);
}

void BSubG::set_chirality_specific_start_range(){
  set_chirality_specific_start_range_base(ptr_gg->n);
}



void ChiralSubG::set_start_range(){
  
  start_range[nsHP::PAD] = {1,2};
  start_range[nsHP::PLU] = {nsHP::no, nsHP::yes};  
  start_range[nsHP::LIW] = {nsHP::no};  
  start_range[nsHP::MIW] = {nsHP::yes};
  start_range[nsHP::WOS] = {0,1,2};

  set_chirality_specific_start_range();

}


void SubG::apply_constraints(){
  for (unsigned hpi = 0; hpi < nHPs; ++hpi){
    if (constraints.at(hpi) != nsHP::undefined){
      
      if (std::find(range[hpi].begin(), range[hpi].end(), constraints.at(hpi)) == range[hpi].end()){
        std::stringstream errm;
        errm << "the constraint on " << Keys[hpi] << " of " << constraints.at(hpi) << " is not in the pre-constraint range:  \n" << get_range_string(hpi);
        errm << "this is not currently allowed";
        throw tinygemm_error(errm.str());
      }
      
      /* we don't worry if it's not in start_range */
      
      

      edges[hpi] = { {constraints.at(hpi), {} } };
      range[hpi] = { constraints.at(hpi) };
      start_range[hpi] = { constraints.at(hpi) };
    }
  }
}

void ChiralSubG::set_preconstraint_edges(){
  
  edges[nsHP::MIC] =
  { {1, {2,3} },
    {2, {1,3,4} },
    {3, {1,2,4} },
    {4, {2,3,5,6} },
    {5, {2,4,6} },
    {6, {4,5,8} },
    {8, {4,6} }    };
    
  edges[nsHP::PAD] = 
  { {0, {1}   },
    {1, {0, 2}}, 
    {2, {1},  }     };
  
  edges[nsHP::PLU] = 
  {  graph_binary  };


  edges[nsHP::LIW] = 
  {  graph_binary  };


  edges[nsHP::MIW] = 
  {  graph_binary  };

  
  /* TODO : add edges */
  /* TODO : namespace the copy types */
  edges[nsHP::WOS] = 
  {  {0, {}},
     {1, {}},
     {2, {}}
   };
  

}


void CSubG::set_start_range(){

  start_range[nsHP::UNR]= {8, 16};
  start_range[nsHP::NAW]= {16, 64};
  start_range[nsHP::GAL]= {nsGAL::byrow, nsGAL::bycol, nsGAL::sucol};
  
  
  /* TODO : put this inside an if on geom and gpu */
  start_range[nsHP::MAC] = {64, 256};
  start_range[nsHP::SKW] = {10};
  
  start_range[nsHP::ICE] = {1};
  start_range[nsHP::PUN] = {nsHP::no, nsHP::yes};
  start_range[nsHP::UFO] = {nsHP::no};

}

void CSubG::set_preconstraint_edges(){

  
  edges[nsHP::UNR] = 
  { {8, {16} },
    {16, {8,32} },
    {32, {16, 64} },
    {64, {16, 32} }  };
  
  edges[nsHP::NAW] = 
  { {64, {16} },
    {16, {64} }  };
  
  edges[nsHP::GAL] = 
  { {nsGAL::byrow, {nsGAL::bycol, nsGAL::sucol}   },
    {nsGAL::bycol, {nsGAL::byrow, nsGAL::sucol}   },
    {nsGAL::sucol, {nsGAL::byrow, nsGAL::bycol}   }   };

  
  /* TODO : put this inside an if on geom and gpu */
  edges[nsHP::MAC] = 
  {
    //{32, {}},
    {64, {256}},
    {256, {64}},    
  };
  
  edges[nsHP::SKW] = 
  {
    {10, {}},
  };
  
  
  edges[nsHP::ICE] = 
  { {1,  {2}},
    {2,  {1,3,4}},
    {3,  {1,2,4,6}},
    {4,  {1,3,5,7}},
    {5,  {1,2,4,6,8}},
    {6,  {1,3,5,7,9}},
    {7,  {4,6,8,10}},
    {8,  {1,5,7,9,11}},
    {9,  {6,8,10,12}},
    {10, {1,7,9,11,13}},
    {11, {8,10,12,14}},
    {12, {1,9,11,13,14}},
    {13, {10,12,14}},
    {14, {1,11,13}}   };

  edges[nsHP::PUN] = 
  {  graph_binary  };
  
  edges[nsHP::UFO] =
  {  graph_binary  };
  
}






void HyperParams::checks() const{
  for (unsigned gi = 0; gi < nsHP::nMats; ++gi){
    if (gi > v_xhps.size()){
      throw tinygemm_error("strange error : gi > v_xhps.size()");
    }
    
    const XHPs & x = v_xhps[gi];
    SubG & graph = *(p_graph->p_subgs[gi]);
    for (unsigned hpi = 0; hpi < graph.nHPs; ++hpi){
      if (hpi >= graph.range.size()){
        std::stringstream errm;
        errm << "strange error : hpi >= graph.range.size()\n";
        errm << "specifically, " << hpi << " >= " << graph.range.size();
        throw tinygemm_error(errm.str());
      }

      auto start = graph.range[hpi].begin();
      auto end = graph.range[hpi].end();
      
      if (x.vs[hpi] == nsHP::undefined || (std::find(start, end, x.vs[hpi]) == end)) {

        std::stringstream errm;
        errm << "\nIn HyperParams::checks(). It appears as though `" << x.vs[hpi] << "' is not a valid value for " << graph.Keys[hpi] << ".\n"; 
        errm << "the relevant graph looks like this: \n" << graph.get_string(hpi);
        throw tinygemm_error(errm.str());
      }
    }
  }
}


void HyperParams::replace(const std::vector<std::vector<unsigned>> & params){
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      v_xhps[mi].vs[hpi] = params.at(mi).at(hpi);
    }
  }
}

/* go through the params, and where it is not nHP::undefined, use its value to replace this */
void HyperParams::replace_where_source_defined(const std::vector<std::vector<unsigned>> & params){
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      if (params[mi][hpi] != nsHP::undefined){
        v_xhps[mi].vs[hpi] = params[mi][hpi];
      }
    }
  }
}

void HyperParams::replace_undefined_randomly(){
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      if (v_xhps[mi].vs[hpi] == nsHP::undefined){
        auto & a_range = p_graph->p_subgs[mi]->start_range[hpi];
        unsigned index = radu.get_from_range (a_range.size());
        v_xhps[mi].vs[hpi] = a_range[index];
      }
    }
  }
}




HyperParams::HyperParams(const Graph & graph):p_graph(&graph) {
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    v_xhps.emplace_back (  XHPs ( p_graph->p_subgs[mi]->nHPs  )  );
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      auto & a_range = p_graph->p_subgs[mi]->start_range[hpi];
      unsigned index = radu.get_from_range (a_range.size());
      v_xhps[mi].vs[hpi] = a_range[index];
    }
  }
  checks();
}

  
bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string(); 
}


std::string HyperParams::get_part_string(char X) const{
  unsigned mi = p_graph->graphind.at(X);
  std::stringstream ss;
  ss << X;
  for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
    ss << "_" << p_graph->p_subgs[mi]->Keys[hpi] << v_xhps[mi].vs[hpi];
  }
  return ss.str();
}

std::string HyperParams::get_string() const{
  std::stringstream ss;
  ss << get_part_string('A') << "__" << get_part_string('B') << "__" << get_part_string('C');  
  return ss.str();
}

std::vector<HyperParams> HyperParams::get_one_aways(){
  
  
  std::vector<HyperParams> one_aways;
  
  /* by changing just one hyper-parameter */
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      unsigned value = v_xhps[mi].vs[hpi];
      for (auto & newval : p_graph->p_subgs[mi]->edges[hpi].at(value)){
        HyperParams hp(*this);
        hp.v_xhps[mi].vs[hpi] = newval;
        one_aways.push_back(hp);        
      }
    }
  }
  
  
  /* by changing MAC and one or both MICs, so as to semi-preserve the overall shape of the macro tile */
  unsigned curr_mac = v_xhps[nsHP::matC].vs[nsHP::MAC];
  for (auto & newmac : p_graph->p_subgs[nsHP::matC]->edges[nsHP::MAC].at(curr_mac)){

    /* ratios of new to current tile grid sizes */
    
    
    auto curr_grid_size = nsMAC::get_mac_grid(curr_mac, v_xhps[nsHP::matC].vs[nsHP::SKW]);
    auto new_grid_size = nsMAC::get_mac_grid(newmac, v_xhps[nsHP::matC].vs[nsHP::SKW]);
    
    double delta_na = static_cast<double>(new_grid_size[nsHP::matA]) / static_cast<double>(curr_grid_size[nsHP::matA]);
    double delta_nb = static_cast<double>(new_grid_size[nsHP::matB]) / static_cast<double>(curr_grid_size[nsHP::matB]);
    
    /* mica scaled so that the macro tile remains ~ the same in the a dimension */
    unsigned curr_mica = v_xhps[nsHP::matA].vs[nsHP::MIC];
    unsigned new_mica = static_cast<unsigned> ( static_cast<double> ( curr_mica ) / delta_na );

    /* micb scaled so that the macro tile remains the same in the b dimension */
    unsigned curr_micb = v_xhps[nsHP::matB].vs[nsHP::MIC];
    unsigned new_micb = static_cast<unsigned> ( static_cast<double> ( curr_micb ) / delta_nb );
    
    /* if the new micro tile (a) is different and valid, add it */
    if (new_mica != curr_mica && in_graph(nsHP::matA, nsHP::MIC, new_mica)){
      HyperParams hp(*this);    
      hp.v_xhps[nsHP::matC].vs[nsHP::MAC] = newmac;
      hp.v_xhps[nsHP::matA].vs[nsHP::MIC] = new_mica;
      one_aways.push_back(hp);
    }

    if (new_micb != curr_micb && in_graph(nsHP::matB, nsHP::MIC, new_micb)){
      HyperParams hp(*this);    
      hp.v_xhps[nsHP::matC].vs[nsHP::MAC] = newmac;
      hp.v_xhps[nsHP::matB].vs[nsHP::MIC] = new_micb;
      one_aways.push_back(hp);

      if (new_mica != curr_mica && in_graph(nsHP::matA, nsHP::MIC, new_mica)){
        HyperParams hp2(hp);
        hp2.v_xhps[nsHP::matA].vs[nsHP::MIC] = new_mica;
        one_aways.push_back(hp2);
      }
    }
  }
  

  unsigned n_uncoupled = one_aways.size();  
  
  
  
  /* by changing two hyper-parameters */  
  for (auto & couple_p : p_graph->coupled_parameters){

    auto first = std::get<0>(couple_p);
    auto first_m = std::get<0>(first);
    auto first_p = std::get<1>(first);
    auto first_value = v_xhps[first_m].vs[first_p];
    
    auto second = std::get<1>(couple_p);
    auto second_m = std::get<0>(second);
    auto second_p = std::get<1>(second);
    auto second_value = v_xhps[second_m].vs[second_p];

    for (auto & new_first_val : p_graph->p_subgs[first_m]->edges[first_p].at(first_value)){
      for (auto & new_second_val : p_graph->p_subgs[second_m]->edges[second_p].at(second_value)){
        
        /* only if one increases and one decreases */
        if ((new_second_val > second_value) != (new_first_val > first_value)){
          HyperParams hp(*this);        
          hp.v_xhps[first_m].vs[first_p] = new_first_val;
          hp.v_xhps[second_m].vs[second_p] = new_second_val;
          one_aways.push_back(hp);
        }
      }
    }
  }
  
  unsigned n_total = one_aways.size();

  /* shuffle them, which bounds the expected time to finding an improvement 
   * (prevents pathological case of all improving kernels at end of vector)  */

  /* shuffle the true one aways */
  radu.shuffle(0, n_uncoupled, one_aways);
  
  
  /* shuffle the two aways (coupled) */
  radu.shuffle(n_uncoupled, n_total, one_aways);
  
  /* shuffle the custom kernels. What? Custom kernels? */
  

  return one_aways;
}
  

     
HyperParams get_hp_start(FindStartType fst, const Graph & graph){

  HyperParams hyper_param_start(graph);

  if (fst == FindStartType::Default){
    std::stringstream ss;
    ss << "getting default in get_hp_start not enabled. Also small matrices will break tinygemm (again) default should return something like\n";
    ss << "A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC5  here.\n";
    throw tinygemm_error(ss.str());
  }

     
  hyper_param_start.checks();  
  return hyper_param_start;
}



bool HyperParams::in_graph(unsigned mi, unsigned hpi, unsigned value){
  return std::count(p_graph->p_subgs[mi]->range[hpi].begin(), p_graph->p_subgs[mi]->range[hpi].end(), value) != 0;
}


bool HyperParams::in_graph(){
  /* filtering out if violates the constraint string */
  bool constraints_satisfied = true;
  for (unsigned mi = 0; mi < nsHP::nMats; ++mi){
    for (unsigned hpi = 0; hpi < p_graph->p_subgs[mi]->nHPs; ++hpi){
      if (in_graph(mi, hpi, v_xhps[mi].vs[hpi]) == false){
      
      //std::count(p_graph->p_subgs[mi]->range[hpi].begin(), p_graph->p_subgs[mi]->range[hpi].end(), v_xhps[mi].vs[hpi]) == 0){
        constraints_satisfied = false;
        break;
      }
    }
  }
  return constraints_satisfied;
}


nsHP::eMat HyperParams::get_eMat_from_char(char X) const{
  X = (X == 'a' ? 'A' : X);
  X = (X == 'b' ? 'B' : X); 
  X = (X == 'c' ? 'C' : X);
  
  if (X != 'A' && X != 'B' && X != 'C'){
    throw tinygemm_error("Problem converting X (char) to nsHP::eMat enumerated type in get_eMat_from_char : " + std::to_string(X) );
  }
  return static_cast<nsHP::eMat> (p_graph->graphind.at(X));
} 

} 
}

