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
  void shuffle(T & t){
    std::shuffle(t.begin(), t.end(), gen);
  }
  
  
};

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

 
SUHP::SUHP(){

  ChiralKeys.resize(nsHP::nChiralHPs);
  NonChiralKeys.resize(nsHP::nNonChiralKeys);
  MatKeys.resize(nsHP::nMatrices);
  HPVals.resize(nsHP::nMatrices);
  HPKeys.resize(nsHP::nMatrices);
  nHPs.resize(nsHP::nMatrices);  
  
  
  ChiralKeys[nsHP::MIC] = "MIC";
  ChiralKeys[nsHP::PAD] = "PAD";
  ChiralKeys[nsHP::PLU] = "PLU";
  ChiralKeys[nsHP::LIW] = "LIW";
  ChiralKeys[nsHP::MIW] = "MIW";
  ChiralKeys[nsHP::WOS] = "WOS";
  ChiralVals = getVals(nsHP::nChiralHPs, ChiralKeys, "ChiralKeys");


  NonChiralKeys[nsHP::UNR] = "UNR";
  NonChiralKeys[nsHP::GAL] = "GAL";
  NonChiralKeys[nsHP::PUN] = "PUN";
  NonChiralKeys[nsHP::ICE] = "ICE";
  NonChiralKeys[nsHP::NAW] = "NAW";
  NonChiralKeys[nsHP::UFO] = "UFO"; 
  NonChiralKeys[nsHP::MAC] = "MAC"; 
  NonChiralVals = getVals(nsHP::nNonChiralKeys, NonChiralKeys, "NonChiralKeys");
  

  MatKeys[nsHP::matA] = 'A';
  MatKeys[nsHP::matB] = 'B';
  MatKeys[nsHP::matC] = 'C';
  MatVals = getVals(nsHP::nMatrices, MatKeys, "MatKeys");
  
  HPVals[nsHP::matA] = ChiralVals;
  HPVals[nsHP::matB] = ChiralVals;
  HPVals[nsHP::matC] = NonChiralVals;


  HPKeys[nsHP::matA] = ChiralKeys;
  HPKeys[nsHP::matB] = ChiralKeys;
  HPKeys[nsHP::matC] = NonChiralKeys;


  nHPs[nsHP::matA] = nsHP::nChiralHPs;
  nHPs[nsHP::matB] = nsHP::nChiralHPs;
  nHPs[nsHP::matC] = nsHP::nNonChiralKeys;  





}

const SUHP suHP;

}

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
  hpg.graph.resize(nsHP::nChiralHPs);
  
  hpg.graph[nsHP::MIC] =
  { {1, {2,3} },
    {2, {1,3,4} },
    {3, {1,2,4} },
    {4, {2,3,5,6} },
    {5, {2,4,6} },
    {6, {4,5,8} },
    {8, {4,6} }    };
  
  hpg.graph[nsHP::PAD] = 
  { {0, {1}   },
    {1, {0}   }     };
  
  hpg.graph[nsHP::PLU] = 
  {  graph_binary  };

  hpg.graph[nsHP::LIW] = 
  {  graph_binary  };

  hpg.graph[nsHP::MIW] = 
  {  graph_binary  };
  
  hpg.graph[nsHP::WOS] = 
  {  {0, {}}   };// for now, no copying TODO(jn) incorporate
  
  hpg.update_range();
  return hpg;
}

Graph get_g_non_chiral(){

  Graph hpg;
  hpg.graph.resize(nsHP::nNonChiralKeys);
  
  hpg.graph[nsHP::UNR] = 
  { {8, {16} },
    {16, {8,32} },
    {32, {16, 64} },
    {64, {16, 32} }  };
  
  hpg.graph[nsHP::NAW] = 
  { {64, {} }  };
  
  hpg.graph[nsHP::GAL] = 
  { {nsGAL::byrow, {nsGAL::bycol, nsGAL::sucol}   },
    {nsGAL::bycol, {nsGAL::byrow, nsGAL::sucol}   },
    {nsGAL::sucol, {nsGAL::byrow, nsGAL::bycol}   }   };

  hpg.graph[nsHP::MAC] = 
  {
    // to be added foe 32 in work group GPUs
    //{nsMAC::a4b8, {nsMAC::a8b8}},
    //{nsMAC::a8b4, {nsMAC::a8b8}},

    //{nsMAC::a8b16, {nsMAC::a8b8, nsMAC::a16b16}},
    //{nsMAC::a16b8, {nsMAC::a8b8, nsMAC::a16b16}},

    
    {nsMAC::a8b8, {nsMAC::a16b16}},
    {nsMAC::a16b16, {nsMAC::a8b8}},
  };
  
  hpg.graph[nsHP::ICE] = 
  { {1,  {2}},
    {2,  {1,3,4}},
    {3,  {1,2,4,6}},
    {4,  {1,3,5,7}},
    {5,  {2,4,6,8}},
    {6,  {3,5,7,9}},
    {7,  {4,6,8,10}},
    {8,  {5,7,9,11}},
    {9,  {6,8,10,12}},
    {10, {7,9,11,13}},
    {11, {8,10,12,14}},
    {12, {9,11,13,14}},
    {13, {10,12,14}},
    {14, {11,13}}   };
  
  hpg.graph[nsHP::PUN] = 
  {  graph_binary  };
  
  hpg.graph[nsHP::UFO] =
  {  graph_binary  };

  hpg.update_range();
  return hpg;
}

Graph g_chiral(get_g_chiral());
Graph g_non_chiral(get_g_non_chiral());

std::vector<std::vector<unsigned>> get_params_from_string(const std::string & hyperstring, bool expect_full_hyperstring){

  std::cout << "in get_params_from_string" << std::endl;

  /* TODO only generate this when an error emerges */
  std::stringstream ssghe;
  ssghe << "the " << (expect_full_hyperstring == true ? "`full'" : "`partial'") << " hyperstring received here is :\n";
  ssghe << "         " << hyperstring << "\n";
  ssghe << "an example of a full hyperstring correctly formated is:\n";
  ssghe << "         ";
  ssghe << "A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC2\n";
  std::string generic_hyperstring_errm = ssghe.str();

  std::string shortkey;
  unsigned val;  

  /* set params to be of the correct shape, filled with nsHP::undefined */
  std::vector<std::vector<unsigned>> params (nsHP::nMatrices);
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    params[mi].resize(suHP.nHPs[mi]);
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      params[mi][hpi] = nsHP::undefined;
    }
  }
  
  auto megafrags = stringutil::split(hyperstring, "__");
  for (auto & megafrag : megafrags){
    char matrixkey = megafrag[0];

    if (matrixkey != 'A' && matrixkey != 'B' && matrixkey != 'C'){
      std::stringstream ss;
      ss << "\nWhile reading hyperstring in get_params_from_string. `matrixkey' should be A,B or C, not `" << matrixkey << "'.\n";
      ss << generic_hyperstring_errm;
      throw tinygemm_error(ss.str());
    }

    auto matrix_val = suHP.MatVals.at(matrixkey);
    auto keyvalfrags = stringutil::split(megafrag.substr(2), "_");
    if (expect_full_hyperstring && (keyvalfrags.size() < suHP.HPKeys[matrix_val].size())){
      std::stringstream ss;
      ss << "While processing frag section " << suHP.MatKeys[matrix_val] << ".\n";
      ss << "There appear to be too few parameters (" << keyvalfrags.size() << " as opposed to " << suHP.HPKeys[matrix_val].size() << ")";
      ss << generic_hyperstring_errm;
      throw tinygemm_error(ss.str());
    }
    
    for (auto & x : keyvalfrags){
      std::tie(shortkey, val) = stringutil::splitnumeric(x);
      auto start = suHP.HPKeys[matrix_val].begin();
      auto end  = suHP.HPKeys[matrix_val].end();
      if(std::find(start, end, shortkey) == end) {
        std::stringstream ss;
        ss << "While processing frag section " << suHP.MatKeys[matrix_val] << ".\n";
        ss << "in get_params_from_string in hyperparams.cpp   :   unrecognised : " + shortkey << ".\n";
        ss << generic_hyperstring_errm;
        throw tinygemm_error(ss.str());
      }
      /* We have confirmed that shortkey is valid, this is safe */
      auto shortkey_val = suHP.HPVals[matrix_val].at(shortkey);
      if (shortkey_val < params[matrix_val].size()){
        params[matrix_val][shortkey_val] = val;
      }
      else{
        throw tinygemm_error("in get_params_from_string, cannot insert as out of bounds.");
      }
    }
  }
  return params;
}

std::string XHPs::get_string() const{
  std::stringstream ss;
  for (unsigned hpi = 0; hpi < vs.size() - 1; ++hpi){
    ss << (*ptr_hpkeys)[hpi] << vs[hpi] << "_";
  }
  ss << (*ptr_hpkeys).back() << vs.back();
  return ss.str();
}

void XHPs::check() const{
  for (unsigned i = 0; i < vs.size(); ++i){
    auto start = ptr_hpgraph->range[i].begin();
    auto end = ptr_hpgraph->range[i].end();
    if (vs[i] == nsHP::undefined || (std::find(start, end, vs[i]) == end)) {
      std::stringstream errm;
      errm << "\nIn XHPs::check(). It appears as though `" << vs[i] << "' is not a valid value for " << (*ptr_hpkeys)[i] << ".\n"; 
      errm << "The valid values are,\n         [";
      for (auto & x : ptr_hpgraph->range[i]){
        errm << " " <<  x << " ";
      }
      errm << "]\n";
      throw tinygemm_error(errm.str());
    }
  }
}

void HyperParams::checks() const{
  for (auto & x : v_xhps){
    x.check();
  }
}


void HyperParams::update(const std::vector<std::vector<unsigned>> & params){
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      v_xhps[mi].vs[hpi] = params.at(mi).at(hpi);
    }
  }
}

HyperParams::HyperParams():
v_xhps {{&suHP.ChiralKeys, &g_chiral, nsHP::nChiralHPs}, 
{&suHP.ChiralKeys, &g_chiral, nsHP::nChiralHPs}, 
{&suHP.NonChiralKeys, &g_non_chiral, nsHP::nNonChiralKeys}} 
{
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      v_xhps[mi].vs[hpi] = nsHP::undefined;
    }
  }
}  

HyperParams::HyperParams(const std::vector<std::vector<unsigned>> & params):HyperParams()
{
  update(params);
  checks();
}


HyperParams::HyperParams(const std::string & hyperstring):HyperParams(get_params_from_string(hyperstring, true)){}


void HyperParams::replace_undefined_randomly(){
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      if (v_xhps[mi].vs[hpi] == nsHP::undefined){
        //select randomly from ptr_hpgraph->range. 
        
        
        
        unsigned index = radu.get_from_range (v_xhps[mi].ptr_hpgraph->range[hpi].size());
        v_xhps[mi].vs[hpi] = v_xhps[mi].ptr_hpgraph->range[hpi][index];
      }
    }
  }
}


HyperParams get_random(std::string constraint_string){
  HyperParams hp;  
  auto constraint_params = get_params_from_string(constraint_string, false);
  hp.update(constraint_params);
  hp.replace_undefined_randomly();
  
  hp.checks();  
  return hp;
}


HyperParams get_cacheless_default(const tinygemm::TinyGemmGeometry & gg, std::string constraint_string){
  HyperParams hp("A_MIC8_PAD1_PLU0_LIW0_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR16_GAL3_PUN0_ICE1_NAW64_UFO0_MAC5");
  auto constraint_params = get_params_from_string(constraint_string, false);
  hp.update(constraint_params);
  return hp;
}


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, std::string constraint_string){
  /* TODO : rather get default from cache, based on gg (?) */
  auto hp = get_cacheless_default(gg, constraint_string);
  throw tinygemm_error("get_default not yet enabled (not needed for paper)");
  return hp;  
}
  
bool HyperParams::operator == (const HyperParams & hpr){
  return get_string() == hpr.get_string(); 
}

std::string HyperParams::get_string() const{
  std::stringstream ss;
  //for (char X : {'A', 'B'}){
  for (unsigned mi : {nsHP::matA, nsHP::matB}){
    ss << suHP.MatKeys[mi] << "_" << v_xhps[mi].get_string() << "__";
  }
  ss << 'C' << "_" << v_xhps[nsHP::matC].get_string();
  return ss.str();
}

std::vector<HyperParams> HyperParams::get_one_aways(){ //const tinygemm::TinyGemmGeometry & gg){
  std::vector<HyperParams> one_aways;
  for (unsigned mi = 0; mi < nsHP::nMatrices; ++mi){
    for (unsigned hpi = 0; hpi < suHP.nHPs[mi]; ++hpi){
      unsigned value = v_xhps[mi].vs[hpi];
      for (auto & newval : v_xhps[mi].ptr_hpgraph->graph[hpi].at(value)){
        HyperParams hp(*this);
        hp.v_xhps[mi].vs[hpi] = newval;
        one_aways.push_back(hp);        
      }
    }
  }

  /* shuffle them, which bounds the expected time to finding an improvement 
   * (prevents pathological case of all improving kernels at end of vector) 
   * currently, we shuffle after adding custom edges, might consider shuffling
   * before adding, to prevent getting trapped in previously found minima.*/

  radu.shuffle(one_aways);
  
  
  
  return one_aways;
}
  

//std::make_tuple(100, 32, 26939, 26939, 26939, 100, true, false), 
/* see dev/python/deepbench/deepbench_results.py : this is generated by get_kernel_cache_string, based on results running find with allotted_time =  30 seconds per problem, with three starting kernels for
 * small, medium, large: On a Fiji! 
 * TODO : regenerate with longer runs and more problems.
 * TODO : should not be a single vector, this has linear find time. At least seperate out isColMajor, tA, tB  
 * TODO : figure out how to make cache contain only reduced problems .... very important! */


    
     
  HyperParams get_hp_start(std::string start_string, std::string constraint_string, const tinygemm::TinyGemmGeometry & gg){
    
    /* we initialise the `hyper-front' with a single HyperParams, */
    /* selected based on problem dimension, constraints and start type  */    
    HyperParams hyper_param_start;


    if (gg.m < 8 || gg.n < 8){
      mowri << "really skinny/thin matrix, returning a default TinyGemmSolution based on gg and constraint_string without searching/benching " << Endl;
      throw std::tinygemm_error("sort this out");
    }
          


    if (start_string.compare("default") == 0){
      hyper_param_start = get_default(gg, constraint_string);
    }
    
    else if (start_string.compare("random") == 0){
      hyper_param_start = get_random(constraint_string);
    }
    
    else if (start_string == ""){
      throw tinygemm_error("start_string should not be empty string");
    }
    
    else {
      /* assume it is a valid hyperstring */
      hyper_param_start = HyperParams(start_string);
    }
    
    // TODO I MUST guarantee that start_hp is going to pass ... warm up bench run with it ?
    
    return hyper_param_start;
  }


} 
}
