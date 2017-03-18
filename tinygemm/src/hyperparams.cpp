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
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/mapkeycheck.hpp>


namespace tinygemm{
namespace hyperparams{

const std::map<unsigned, std::vector<unsigned> > graph_binary = 
{   {0, {1}},
    {1, {0}}    };

void Graph::update_range(){
  range.resize(graph.size());
  for (unsigned i = 0; i < graph.size(); ++i){
    for (auto & x : graph[i]){
      range[i].push_back(x.first);
    }
  }
}

Graph get_g_chiral(){

  Graph hpg;
  hpg.graph.resize(nChiralHPs);
  
  hpg.graph[MIC] =
  { {1, {2,3,4} },
    {2, {1,3,4} },
    {3, {1,2,4,5} },
    {4, {1,2,3,5,6} },
    {5, {2,3,4,6,7} },
    {6, {3,4,5,7,8} },
    {7, {4,5,6,8} },
    {8, {4,6,7} }    };
  
  hpg.graph[PAD] = 
  { {0, {1,2}   },
    {1, {0,2}   },
    {2, {0,1}   }    };
  
  hpg.graph[PLU] = 
  {  graph_binary  };

  hpg.graph[LIW] = 
  {  graph_binary  };

  hpg.graph[MIW] = 
  {  graph_binary  };
  
  hpg.graph[WOS] = 
  {  {0, {}}   };// for now, no copying TODO
  
  hpg.update_range();
  return hpg;
}

Graph get_g_non_chiral(){

  Graph hpg;
  hpg.graph.resize(nNonChiralHPs);
  
  hpg.graph[UNR] = 
  { {4, {8} },
    {8, {4,16} },
    {16, {8,24,32} },
    {24, {16, 32} },
    {32, {16} }  };
  
  hpg.graph[NAW] = 
  { {16, {81}},
    {81, {16}}   };
  
  hpg.graph[GAL] = 
  { {byrow, {bycol, sucol}   },
    {bycol, {byrow, sucol}   },
    {sucol, {byrow, bycol}   }   };

  hpg.graph[MAC] = 
  { {a4b8, {a8b8}},
    {a8b4, {a8b8}},
    {a8b8, {a4b8, a8b4, a8b8, a8b16, a16b8, a16b16}},
    {a8b16, {a8b8, a16b16}},
    {a16b8, {a8b8, a16b16}}    };
  
  hpg.graph[ICE] = 
  { {1,  {2,3}},
    {2,  {1,3,4}},
    {3,  {1,2,4,6}},
    {4,  {1,2,3,5,6,7}},
    {5,  {2,3,4,6,7,8}},
    {6,  {3,4,5,7,8,9}},
    {7,  {4,5,6,8,9,10}},
    {8,  {5,6,7,9,10,11}},
    {9,  {6,7,8,10,11,12}},
    {10, {7,8,9,11,12,13}},
    {11, {8,9,10,12,13,14}},
    {12, {9,10,11,13,14}},
    {13, {10,11,12,14}},
    {14, {11,12,13}}   };
  
  hpg.graph[PUN] = 
  {  graph_binary  };
  
  hpg.graph[UFO] =
  {  graph_binary  };

  hpg.update_range();
  return hpg;
}

Graph g_chiral(get_g_chiral());
Graph g_non_chiral(get_g_non_chiral());

/* take in hyper-parameter string and return a map */
std::map<char, std::map<std::string, unsigned> > get_params_from_string(const std::string & hyperstring){
  
  std::string shortkey;
  unsigned val;
  
  std::map<char, std::map<std::string, unsigned> > params = { {'A', {}}, {'B', {}}, {'C', {}} }; 
  auto megafrags = stringutil::split(hyperstring, "__");
  for (auto & megafrag : megafrags){
    char matrixkey = megafrag[0];
    if (matrixkey != 'A' && matrixkey != 'B' && matrixkey != 'C'){
      throw tinygemm_error("matrixkey should be A,B or C");
    }
    auto keyvalfrags = stringutil::split(megafrag.substr(2), "_");
    for (auto & x : keyvalfrags){
      std::tie(shortkey, val) = stringutil::splitnumeric(x);
      params[matrixkey][shortkey] = val;
    }
  }
  return params;
}


void XHPs::check() const{
  for (unsigned i = 0; i < vs.size(); ++i){
    auto start = hpgraph.range[i].begin();
    auto end = hpgraph.range[i].end();
    if(std::find(start, end, vs[i]) == end) {
      std::stringstream errm;
      errm << "\nIn XHPs::check(). It appears as though `" << vs[i] << "' is not a valid value for " << hpkeys[i] << ".\n"; 
      throw tinygemm_error(errm.str());
    }
  }
}

void HyperParams::checks() const{
  for (auto & x : v_xhps){
    x.check();
  }
}

void HyperParams::check_map_keys(const std::map<char, std::map<std::string, unsigned> > & params) const{  
  for (char X : {'A', 'B', 'C'}){
    mapkeycheck::check_map_keys(
    params.at(X), 
    HPKeys.at(X), 
    std::string("HyperParams constructor, params against keys, ") + X);
  }
}
    

HyperParams::HyperParams(const std::map<char, std::map<std::string, unsigned> > & params){
  check_map_keys(params);
  for (char X : {'A', 'B', 'C'}){
    for (unsigned i = 0; i < nHPs[X]; ++i){
      v_xhps[matNums.at(X)].vs[i] = params.at(X).at(HPKeys.at(X)[i]);
    }
  }
  checks();
}

HyperParams::HyperParams(const std::string & hyperstring):HyperParams(get_params_from_string(hyperstring)){}

/* Find the nearest geometry in the cache, and take its hyper params */
HyperParams get_default(){
  return std::string("A_MAC64_MIC4_PAD0_PLU1_LIW1_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW1_MIW1_WOS0__U8_GA3_PU1_ICE7_NAW64_UFO0");
}
  
bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string(); 
}


std::string HyperParams::get_string() const{
  std::stringstream ss;
  for (char X : {'A', 'B'}){
    ss << X << "_" << v_xhps[matNums.at(X)].get_string() << "__";
  }
  ss << 'C' << "_" << v_xhps[matNums.at('C')].get_string();
  return ss.str();
}
    
void add_hyperparam(const std::string & hyperstring, std::vector<HyperParams> & one_aways){
  one_aways.push_back(HyperParams(hyperstring));
}

std::vector<HyperParams> HyperParams::get_one_aways(){
  
  std::vector<HyperParams> one_aways;
  
  for (auto X   : {'A', 'B', 'C'} ){
    for (unsigned hp_i = 0; hp_i < HPKeys.at(X).size(); ++hp_i){
      unsigned value = v_xhps.at(matNums.at(X)).vs[hp_i];
      for (auto & newval : v_xhps.at(matNums.at(X)).hpgraph.graph[hp_i].at(value)){
        HyperParams hp(*this);
        hp.v_xhps.at(X).vs[hp_i] = newval;
      }
    }
  }

  /* shuffle them, which bounds the expected time to finding an improvement 
   * (prevents pathological case of all improving kernels at end of vector) 
   * currently, we shuffle after adding custom edges, might consider shuffling
   * before adding, to prevent getting trapped in previously found minima.*/
  std::random_device rd;
  std::default_random_engine default_engine(rd());
  std::shuffle(one_aways.begin(), one_aways.end(), default_engine);
  return one_aways;
}
  
  




//std::make_tuple(100, 32, 26939, 26939, 26939, 100, true, false), 
/* see dev/python/deepbench/deepbench_results.py : this is generated by get_kernel_cache_string, based on results running find with allotted_time =  30 seconds per problem, with three starting kernels for
 * small, medium, large: On a Fiji! 
 * TODO : regenerate with longer runs and more problems.
 * TODO : should not be a single vector, this has linear find time. At least seperate out isColMajor, tA, tB  
 * TODO : figure out how to make cache contain only reduced problems .... very important! */



}


 
}



