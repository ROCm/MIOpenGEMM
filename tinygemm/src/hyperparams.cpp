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
{
    {0, {1}},
    {1, {0}}
};

void HyperParamsGraph::update_range(){
  range.resize(graph.size());
  for (unsigned i = 0; i < graph.size(); ++i){
    for (auto & x : graph[i]){
      range[i] = x.first;
    }
  }
}





ChiralHyperParamsGraph::ChiralHyperParamsGraph {
  
  ChiralHyperParamsGraph():HyperParamsGraph() {
    
    graph.resize(NumberOfChiralHyperParams);
    
    graph[MIC] =
    {
      {1, {2,3,4} },
      {2, {1,3,4} },
      {3, {1,2,4,5} },
      {4, {1,2,3,5,6} },
      {5, {2,3,4,6,7} },
      {6, {3,4,5,7,8} },
      {7, {4,5,6,8} },
      {8, {4,6,7} }
    };
    
    graph[PAD] = 
    { 
      {0, {1,2}   },
      {1, {0,2}   },
      {2, {0,1}   }
    };
    
    graph[PLU] = 
    {
      graph_binary
    };
  
    graph[LIW] = 
    {
      graph_binary
    };
  
    graph[MIW] = 
    {
      graph_binary
    };
    
    graph[WOS] = 
    {
      {0, {}} // for now, no copying TODO
    };
    
    update_range();
  }
};


NonChiralHyperParamsGraph::NonChiralHyperParamsGraph{
  
  NonChiralHyperParamsGraph():HyperParamsGraph(){
    
    graph.resize(NumberOfNonChiralHyperParams);
    
    graph[UNR] = 
    {
      {4, {8} },
      {8, {4,16} },
      {16, {8,24,32} },
      {24, {16, 32} },
      {32, {16} }
    };
    
    graph[NAW] = 
    { 
      {16, {81}},
      {81, {16}}
    };
    
    graph[GAL] = 
    { 
      {byrow, {bycol, sucol}   },
      {bycol, {byrow, sucol}   },
      {sucol, {byrow, bycol}   }
    };
  
    graph[MAC] = 
    {
      {a4b8, {a8b8}},
      {a8b4, {a8b8}},
      {a8b8, {a4b8, a8b4, a8b8, a8b16, a16b8, a16b16}},
      {a8b16, {a8b8, a16b16}},
      {a16b8, {a8b8, a16b16}}
    };
    
    graph[ICE] = 
    { 
      {1,  {2,3}},
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
      {14, {11,12,13}}
    };
    
    graph[PUN] = 
    { 
      graph_binary
    };
  
    graph[UFO] =
    { 
      graph_binary
    }; 
  
    update_range();
  }
};
  

/* take in hyper-parameter string and return a map */
std::map<char, std::map<std::string, unsigned> > get_params_from_string(const std::string & hyperstring){
  std::map<char, std::map<std::string, unsigned> > params = { {'A', {}}, {'B', {}}, {'C', {}} }; 
  auto megafrags = stringutil::split(hyperstring, "__");
  for (auto & frag : megafrags){
    char matrixkey = frag[0];
    auto keyvalfrags = stringutil::split(megafrags.substr(2), "_");
    for (auto & x : frags){
      std::tie(shortkey, val) = stringutil::splitnumeric(x);
      params[matrixkey][shortkey] = val;
    }
  }
  return params;
}


void XChiralHyperParams::check() const{
  for (unsigned i = 0; i < values.size(); ++i){
    ptr_hpgraph->
  }
    
}

void HyperParams::checks() const{

  aps.checks();
  bps.checks();
  
  
  bool_check("unroll_pragma", unroll_pragma.val);
  bool_check("unroll_for_offset", unroll_for_offset.val);
  positive_check("unroll", unroll.val);
  positive_check("n_target_active_workgroups", n_target_active_workgroups.val);
  positive_check("n_work_items_per_c_elm", n_work_items_per_c_elm.val);
  ga_check();
  
}
    

HyperParams::HyperParams(const std::map<char, std::map<std::string, unsigned> > & params):

{

  check_map_keys(params);
    
  
  for (char X : {'A', 'B'}){
    at(X).micro_tile_length.val = params.at(X).at("MIC");
    at(X).load_pll_to_unroll.val = params.at(X).at("PLU");
    at(X).workspace_type.val = params.at(X).at("WOS");
    at(X).lds_pad_size.val = params.at(X).at("PAD");
    at(X).load_to_lds_interwoven.val = params.at(X).at("LIW");
    at(X).c_micro_tiles_interwoven.val = params.at(X).at("MIW");
  }
 
  unroll.val = params.at('C').at("U");
  group_allocation.val = params.at('C').at("GA");
  unroll_pragma.val = params.at('C').at("PU");
  n_work_items_per_c_elm.val = params.at('C').at("ICE");
  n_target_active_workgroups.val = params.at('C').at("NAW");
  unroll_for_offset.val = params.at('C').at("UFO");
  macro_tile_grid.val = params.at(X).at("MAC"); 

  checks();
}


HyperParams::HyperParams(const std::string & hyperstring):HyperParams(get_params_from_string(hyperstring)){}


  
void HyperParams::check_map_keys(const std::map<char, std::map<std::string, unsigned> > & params){
  for (char X : {'A', 'B'}){
    mapkeycheck::check_map_keys(params.at(X), chiral_pl.keys, std::string("HyperParams constructor, params against keys, ") + X);
  }
  mapkeycheck::check_map_keys(params.at('C'), nonchiral_pl.keys, std::string("HyperParams constructor, params against keys, C"));
}


    

HyperParams get_default_small(bool enforce_deterministic){
  
  std::string ice = std::to_string(enforce_deterministic == false ? 3 : 1);
  return "A_MAC8_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC8_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE" +  ice + "_NAW64_UFO0";
}

HyperParams get_default_tiniest(bool enforce_deterministic){
  
  std::string ice = std::to_string(enforce_deterministic == false ? 3 : 1);
  return "A_MAC1_MIC1_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC1_MIC1_PAD1_PLU1_LIW0_MIW1_WOS0__U16_GA1_PU1_ICE" +  ice + "_NAW64_UFO0";
}



/* Find the nearest geometry in the cache, and take its hyper params */
HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic){
  
                      //A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U16_GA3_PU0_ICE1_NAW64_UFO0
  //return std::string("A_MAC128_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MAC96_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__U8_GA1_PU1_ICE4_NAW64_UFO0");
  
  /* The case  of  (gg.m < 8 || gg.n < 8) */  
  if (gg.m < 8 || gg.n < 8) {
    return get_default_tiniest(enforce_deterministic);
  }

  HyperParams best_hp = get_default_small(enforce_deterministic);
  float min_distance = std::numeric_limits<float>::max();

  for (auto geohyp : HyperParams::kernel_cache){
    tinygemm::TinyGemmGeometry geo = std::get<0>(geohyp);
    std::string hpstring = std::get<1>(geohyp);
    
    float new_distance = gg.get_distance(geo);
    if (new_distance < min_distance){
      best_hp = HyperParams(hpstring);
      min_distance = new_distance;
    }
  }

  if (enforce_deterministic == true){
    best_hp.n_work_items_per_c_elm.val= 1;
  }

      
  return best_hp;
}
  
unsigned HyperParams::get_nwitems_h(){
  return aps.macro_tile_length.val / aps.micro_tile_length.val;
}

unsigned HyperParams::get_nwitems_w(){
  return bps.macro_tile_length.val / bps.micro_tile_length.val;
}

bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string();
}
  
    
void add_hyperparam(const std::string & hyperstring, std::vector<HyperParams> & one_aways){
  one_aways.push_back(HyperParams(hyperstring));
}

  
std::vector<HyperParams> HyperParams::get_one_aways(const tinygemm::TinyGemmGeometry & gg){
  
  std::vector<HyperParams> one_aways;
    
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



};


 
}
}


