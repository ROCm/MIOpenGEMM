#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/mapkeycheck.hpp>


namespace tinygemm{


namespace hyperparams{

/* TODO : move this out and make constructor which takes seed */
class RandomUtil {

private:
  std::random_device rd;
  std::default_random_engine gen;
  std::uniform_int_distribution<unsigned> unidis;

public:
  RandomUtil():rd(), gen(rd()) {} 
  unsigned get_from_range(unsigned upper){
    return unidis(gen) % upper;
  }
  
  template <typename T>
  void shuffle(unsigned start_index, unsigned end_index, T & t){
    if (end_index > t.size() || start_index > end_index){
      throw tinygemm_error("problem in shuffle");
    }
    std::shuffle(t.begin() + start_index, t.begin() + end_index, gen);
  }
};

RandomUtil radu;


Graph::Graph(const tinygemm::TinyGemmGeometry & gg) : ptr_gg(&gg), asubg(gg), bsubg(gg), csubg(gg)  {

  p_subgs.resize(nsHP::nGraph);
  p_subgs[nsHP::SubGChiralA] = &asubg;
  p_subgs[nsHP::SubGChiralB] = &bsubg;
  p_subgs[nsHP::SubGNonChiral] = &csubg;

  graphind['A'] = nsHP::SubGChiralA;
  graphind['B'] = nsHP::SubGChiralB;
  graphind['C'] = nsHP::SubGNonChiral;
  
  coupled_parameters.push_back( { {nsHP::matA, nsHP::MIC}, {nsHP::matB, nsHP::MIC} } );
  coupled_parameters.push_back( { {nsHP::matC, nsHP::UFO}, {nsHP::matC, nsHP::PUN} } );

}






std::vector<std::vector<unsigned>> Graph::get_params_from_string(const std::string & hyperstring, bool expect_full_hyperstring){

  /* TODO only generate this when an error emerges */
  std::stringstream ssghe;
  ssghe << "the " << (expect_full_hyperstring == true ? "`full'" : "`partial'") << " hyperstring received here is :\n";
  ssghe << "         " << hyperstring << "\n";
  ssghe << "an example of a full hyperstring correctly formated is:\n";
  ssghe << "         ";
  ssghe << "A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC2\n";
  std::string generic_hyperstring_errm = ssghe.str();

  /* MIC, etc. */
  std::string key; 
  /* 6, etc */
  unsigned val;  

  /* set params to be of the correct shape, filled with nsHP::undefined */
  std::vector<std::vector<unsigned>> params (nsHP::nGraph);
  for (unsigned mi = 0; mi < nsHP::nGraph; ++mi){
    params[mi] = std::vector<unsigned> (p_subgs[mi]->nHPs, nsHP::undefined);
  }
    
  auto megafrags = stringutil::split(hyperstring, "__");
  
  for (auto & megafrag : megafrags){
    
    char graphchar = megafrag[0];
    if (graphchar != 'A' && graphchar != 'B' && graphchar != 'C'){
      std::stringstream ss;
      ss << "\nWhile reading hyperstring in get-params-from-string. `graphchar' should be A,B or C, not `" << graphchar << "'.\n";
      ss << generic_hyperstring_errm;
      throw tinygemm_error(ss.str());
    }

    unsigned graphnum = graphind[graphchar];
    auto & subg = *(p_subgs[graphnum]);
    auto keyvalfrags = stringutil::split(megafrag.substr(2), "_");
    auto expected_nHPs = subg.nHPs;
    
    if (expect_full_hyperstring && (keyvalfrags.size() < expected_nHPs)){
      std::stringstream ss;
      ss << "While processing frag section " << graphchar << ".\n";
      ss << "There appear to be too few parameters (" << keyvalfrags.size() << " as opposed to " << expected_nHPs << ")";
      ss << generic_hyperstring_errm;
      throw tinygemm_error(ss.str());
    }
    
    for (auto & x : keyvalfrags){
      std::tie(key, val) = stringutil::splitnumeric(x);
      auto start = subg.Keys.begin();
      auto end  = subg.Keys.end();
      if(std::find(start, end, key) == end) {
        std::stringstream ss;
        ss << "While processing frag section " << graphchar << ".\n";
        ss << "in get-params-from-string in hyperparams.cpp.  Unrecognised : " + key << ".\n";
        ss << generic_hyperstring_errm;
        throw tinygemm_error(ss.str());
      }
      /* We have confirmed that key is valid, this is safe */
      auto keyindex = subg.Vals.at(key);
      if (keyindex < params[graphnum].size()){
        params[graphnum][keyindex] = val;
      }
      else{
        throw tinygemm_error("in get-params-from-string, cannot insert as out of bounds.");
      }
    }
  }
  return params;
}


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
  NonChiralVals = getVals(nsHP::nNonChiralHPs, Keys, "NonChiralKeys");
}



const std::map<unsigned, std::vector<unsigned> > graph_binary = 
{   {0, {1}},
    {1, {0}}    };

void SubG::initialise_range_from_edges(){
  range.resize(graph.size());
  for (unsigned i = 0; i < graph.size(); ++i){
    for (auto & x : edges[i]){
      range[i].push_back(x.first);
    }
  }
}




SubG::SubG(unsigned nHPs_, const tinygemm::TinyGemmGeometry & gg): nHPs(nHPs_), ptr_gg(&gg), Keys(size), Vals(size), graph (size), start_range (size), gg_start_range (size) {}

void SubG::initialise(const tinygemm::TinyGemmGeometry & gg){
  initialise_maps();
  set_edges();
  initialise_range_from_edges();
  set_start_range();
  confirm_start_is_subset();
}

void SubG::confirm_start_is_subset(){
  //TODO
}

ChiralSubG::ChiralSubG(const tinygemm::TinyGemmGeometry & gg) : SubG(nsHP::nChiralHPs, gg){}

CSubG::CSubG(const tinygemm::TinyGemmGeometry & gg) : SubG(nsHP::nNonChiralHPs, gg){}

void ASubG::set_chirality_specific(){
  start_range[nsHP::MIC] = {8};  
}

void BSubG::set_chirality_specific(){
  start_range[nsHP::MIC] = {8};  
}



void ChiralSubG::set_start_range(const tinygemm::TinyGemmGeometry & gg){
  
  start_range[nsHP::MIC] = {};
  start_range[nsHP::PAD] = {1};    
  start_range[nsHP::PLU] = {nsHP::no, nsHP::yes};  
  start_range[nsHP::LIW] = {nsHP::no};  
  start_range[nsHP::MIW] = {nsHP::yes};
  start_range[nsHP::WOS] = {0};

  set_chirality_specific();

}

void ChiralSubG::set_edges(const tinygemm::TinyGemmGeometry & gg){
  
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

  
  edges[nsHP::WOS] = 
  {  {0, {}}   };// for now, no copying TODO(jn) incorporate
}


void CSubG::set_start_range(const tinygemm::TinyGemmGeometry & gg){

  //gg currently not used for non-chiral
  (void)gg;
  
  start_range[nsHP::UNR]= {8, 16};
  start_range[nsHP::NAW]= {16, 64};
  start_range[nsHP::GAL]= {nsGAL::byrow, nsGAL::bycol, nsGAL::sucol};
  start_range[nsHP::MAC]= {nsMAC::a8b8, nsMAC::a16b16};
  start_range[nsHP::ICE] = {1};
  start_range[nsHP::PUN] = {nsHP::no, nsHP::yes};
  start_range[nsHP::UFO] = {nsHP::no};

}

void CSubG::set_edges(const tinygemm::TinyGemmGeometry & gg){

  //gg currently not used for non-chiral
  (void)gg;
  
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

  edges[nsHP::MAC] = 
  {
    // to be added foe 32 in work group GPUs
    //{nsMAC::a4b8, {nsMAC::a8b8}},
    //{nsMAC::a8b4, {nsMAC::a8b8}},

    //{nsMAC::a8b16, {nsMAC::a8b8, nsMAC::a16b16}},
    //{nsMAC::a16b8, {nsMAC::a8b8, nsMAC::a16b16}},

    //{nsMAC::a1b1, {nsMAC::a4b4}},
    //{nsMAC::a4b4, {nsMAC::a8b8, nsMAC::a8b8}},
    {nsMAC::a8b8, {nsMAC::a16b16}},
    {nsMAC::a16b16, {nsMAC::a8b8}},
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
  //for (auto & x : v_xhps){
  for (unsigned gi = 0; gi < nsHP::nGraph; ++gi){
    XHPs & x = v_xhps[gi];
    SubG & graph = *(ptr_graphs->ptr_graphs[gi]);
    for (unsigned i = 0; i < graph.nHPs; ++i){
      auto start = graph.range[i].begin();
      auto end = graph.range[i].end();
      if (vs[i] == nsHP::undefined || (std::find(start, end, x.vs[i]) == end)) {
        std::stringstream errm;
        errm << "\nIn XHPs::check(). It appears as though `" << x.vs[i] << "' is not a valid value for " << graph.Keys[i] << ".\n"; 
        errm << "The valid values are,\n         [";
        for (auto & valid_value : graph.range[i]){
          errm << " " <<  valid_value << " ";
        }
        errm << "]\n";
        throw tinygemm_error(errm.str());
      }
    }
  }
}


void HyperParams::replace(const std::vector<std::vector<unsigned>> & params){
  for (unsigned mi = 0; mi < nsHP::nGraph; ++mi){
    for (unsigned hpi = 0; hpi < ptr_graphs->ptr_graphs[mi].nHPs; ++hpi){
      v_xhps[mi].vs[hpi] = params.at(mi).at(hpi);
    }
  }
}

/* go through the params, and where it is not nHP::undefined, use its value to replace this */
void HyperParams::replace_where_source_defined(const std::vector<std::vector<unsigned>> & params){
  for (unsigned mi = 0; mi < nsHP::nGraph; ++mi){
    for (unsigned hpi = 0; hpi < ptr_graphs->ptr_graphs[mi].nHPs[mi]; ++hpi){
      if (params[mi][hpi] != nsHP::undefined){
        v_xhps[mi].vs[hpi] = params[mi][hpi];
      }
    }
  }
}

void HyperParams::replace_undefined_randomly(){
  for (unsigned mi = 0; mi < nsHP::nGraph; ++mi){
    for (unsigned hpi = 0; hpi < ptr_graphs->ptr_graphs[mi].nHPs[mi]; ++hpi){
      if (v_xhps[mi].vs[hpi] == nsHP::undefined){
        auto & a_range = ptr_graphs->ptr_graphs[mi]->start_range[hpi];
        unsigned index = radu.get_from_range (a_range.size());
        v_xhps[mi].vs[hpi] = a_range[index];
      }
    }
  }
}




HyperParams::HyperParams(const Graph & graphs):ptr_graphs(&graphs) {
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    v_xhps.emplace_back {graphs->ptr_graphs.nHPs};
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      v_xhps[mi].vs[hpi] = nsHP::undefined;
    }
  }
}  

HyperParams::HyperParams(const Graph & graphs, const std::vector<std::vector<unsigned>> & params):HyperParams(graphs)
{
  replace(params);
  checks();
}


HyperParams::HyperParams(const Graph & graphs, const std::string & hyperstring): HyperParams(graphs, get_params_from_string(hyperstring, true)){}








  
bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string(); 
}



std::string HyperParams::get_string() const{
  
  
  throw tinygemm_error("can't fo string sm oys");
//std::string XHPs::get_string() const{  
  //std::stringstream ss;
  //for (unsigned hpi = 0; hpi < vs.size() - 1; ++hpi){
    //ss << (*ptr_hpkeys)[hpi] << vs[hpi] << "_";
  //}
  //ss << (*ptr_hpkeys).back() << vs.back();
  //return ss.str();
//}



  //std::stringstream ss;
  //for (unsigned mi : {nsHP::matA, nsHP::matB}){
    //ss << suHP.MatKeys[mi] << "_" << v_xhps[mi].get_string() << "__";
  //}
  //ss << 'C' << "_" << v_xhps[nsHP::matC].get_string();
  //return ss.str();
}

std::vector<HyperParams> HyperParams::get_one_aways(){
  
  std::vector<HyperParams> one_aways;
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < ptr_graphs->ptr_graphs[mi].nHPs; ++hpi){
      unsigned value = v_xhps[mi].vs[hpi];
      for (auto & newval : ptr_hpgraphs.ptr_hpgraphs[mi]->edges[hpi].at(value)){
        HyperParams hp(*this);
        hp.v_xhps[mi].vs[hpi] = newval;
        one_aways.push_back(hp);        
      }
    }
  }
  unsigned n_uncoupled = one_aways.size();
  
  
  for (auto & couple_p : ptr_graphs->coupled_parameters){

    auto first = std::get<0>(couple_p);
    auto first_m = std::get<0>(first);
    auto first_p = std::get<1>(first);
    auto first_value = v_xhps[first_m].vs[first_p];
    
    auto second = std::get<1>(couple_p);
    auto second_m = std::get<0>(second);
    auto second_p = std::get<1>(second);
    auto second_value = v_xhps[second_m].vs[second_p];

    for (auto & new_first_val : ptr_graphs->ptr_graphs[first_m]->edges[first_p].at(first_value)){
      for (auto & new_second_val : ptr_graphs->ptr_graphs[second_m]->edges[second_p].at(second_value)){      
        HyperParams hp(*this);
        
        hp.v_xhps[first_m].vs[first_p] = new_first_val;
        hp.v_xhps[second_m].vs[second_p] = new_second_val;
        one_aways.push_back(hp);
      
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
  

//std::make_tuple(100, 32, 26939, 26939, 26939, 100, true, false), 
/* see dev/python/deepbench/deepbench_results.py : this is generated by get_kernel_cache_string, based on results running find with allotted_time =  30 seconds per problem, with three starting kernels for
 * small, medium, large: On a Fiji! 
 * TODO : regenerate with longer runs and more problems.
 * TODO : should not be a single vector, this has linear find time. At least seperate out isColMajor, tA, tB  
 * TODO : figure out how to make cache contain only reduced problems .... very important! */


     
HyperParams get_hp_start(FindStartType fst, std::string constraint_string, const Graph & graphs){

  auto constraint_params = graphs.get_params_from_string(constraint_string, false);

  HyperParams hyper_param_start(graphs);
  if (fst == FindStartType::Default){
    hyper_param_start = graphs.get_default();
    hyper_param_start.replace_where_source_defined(constraint_params);
  }
  
  else if (fst == FindStartType::Random){
    hyper_param_start.replace(constraint_params);
    hyper_param_start.replace_undefined_randomly(graphs);
  }
  
  hyper_param_start.checks();  
  return hyper_param_start;
}




bool HyperParams::satisfies_where_source_defined(const std::vector<std::vector<unsigned>> & params){
  /* filtering out if violates the constraint string */
  bool constraints_satisfied = true;
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      if (params[mi][hpi] != nsHP::undefined && v_xhps[mi].vs[hpi] != params[mi][hpi]){
        constraints_satisfied = false;
        break;
      }
    }
  }
  return constraints_satisfied;
}

tinygemm::HyperParams get_default(Graph & graph){
  if (graph.ptr_gg->m < 8 || graph.ptr_gg->n < 8){
    std::cout << "really skinny/thin matrix, returning a default TinyGemmSolution based on gg and constraint_string without searching/benching " << std::endl;
    throw tinygemm_error("sort this out");
  }
    
  /* TODO : rather get default from cache, based on gg (?) */
  auto hp = HyperParams hp("A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC5");
  
  throw tinygemm_error("get_default not yet enabled (in paper writing mode)");
  return hp;  
}


} 
}








//SUHP::SUHP(){

  //MatKeys.resize(nsHP::nMatrices);
  //HPVals.resize(nsHP::nMatrices);
  //HPKeys.resize(nsHP::nMatrices);
  


  //MatVals = getVals(nsHP::nMatrices, MatKeys, "MatKeys");
  
  //HPVals[nsHP::matA] = ChiralVals;
  //HPVals[nsHP::matB] = ChiralVals;
  //HPVals[nsHP::matC] = NonChiralVals;


  //HPKeys[nsHP::matA] = ChiralKeys;
  //HPKeys[nsHP::matB] = ChiralKeys;
  //HPKeys[nsHP::matC] = NonChiralKeys;




  ////SubGKeys[nsHP::ChiralType] = ChiralKeys;
  ////SubGKeys[nsHP::NonChiralType] = NonChiralKeys;

//}

//const SUHP suHP;
