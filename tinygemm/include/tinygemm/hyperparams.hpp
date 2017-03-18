#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include "tinygemmgeometry.hpp"

namespace tinygemm{

namespace hyperparams{
  
class HyperParamsGraph{
  /* example : graph[MIC] is a map; graph[MIC][1] --> {2,3,4} */
  const std::vector<std::map<unsigned, std::vector<unsigned> > > graph;
  /* example : range[MIC] --> {1,2,3,4,5,6,7,8} */
  const std::vector<std::vector<unsigned> > range;  
  void update_range();
};

class ChiralHyperParamsGraph:public HyperParamsGraph{
  ChiralHyperParamsGraph();
}

class NonChiralHyperParamsGraph:public HyperParamsGraph{
  ChiralHyperParamsGraph();
}

extern const ChiralHyperParamsGraph * ptr_chiral_hyper_params_graph;
extern const NonChiralHyperParamsGraph * ptr_non_chiral_hyper_params_graph;

enum EnumMAC {a4b8 = 0, a8b4 = 1, a8b8 = 2, a8b16 = 3, a16b8 = 4, a16b16 = 5};
enum EnumGA  {byrow = 1, bycol = 2, sucol = 3};

/* if you're going to add parameter here, make sure to add it BEFORE the final count */
enum EnumChiralHyperParam {MIC = 0, PAD, PLU, LIW, MIW, WOS,          NumberOfChiralHyperParams};
/* the strings must appear in same order as enum */
const std::vector<std::string> ChiralHyperParamShortkeys = {"MIC", "PAD", "PLU", "LIW", "MIW", "WOS"};

enum EnumNonChiralHyperParam {UNR = 0, GAL, PUN, ICE, NAW, UFO, MAC,      NumberOfNonChiralHyperParams};
const std::vector<std::string> NonChiralHyperParamShortkeys = {"UNR", "GAL", "PUN", "ICE", "NAW", "UFO", "MAC"};

// TODO : confirm somewhere that NumberOfChiralHyperParams = chiralHyperParamShortkeys.size() etc
enum EnumMatrix {matA = 0, matB, matC, NumberOfMatrices};

class XChiralHyperParams{
  
  protected:
    const std::vector<std::string> & shortkeys;
    const HyperParamsGraph * ptr_hpgraph;
      
  public:
    std::vector<unsigned> values;
    std::vector<unsigned> importances;
    XChiralHyperParams(unsigned size):values(size), importances(size) {}
    std::string get_string() const;
    void check() const;
    
    XChiralHyperParams(const std::vector<std::string> & shortkeys_, const HyperParamsGraph * ptr_hpgraph_, unsigned MParamsTest)
    
};


class HyperParams{

private:
  XChiralHyperParams aps(ChiralHyperParamShortkeys, .., NumberOfChiralHyperParams);
  XChiralHyperParams bps(ChiralHyperParamShortkeys, .., NumberOfChiralHyperParams);
  XChiralHyperParams cps(NonChiralHyperParamShortkeys, .., NumberOfNonChiralHyperParams);

  std::vector<XChiralHyperParams> xchiralhps;

public:

  const XChiralHyperParams & at(EnumMatrix matX) const {return  xchiralhps[matX]; }
  XChiralHyperParams & at(EnumMatrix matX) {return  xchiralhps[matX]; }
  
  HyperParams(const std::map<char, std::map<std::string, unsigned> > & );

  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  
  std::vector<HyperParams> get_one_aways(const tinygemm::TinyGemmGeometry & gg);

  std::map<char, std::map<std::string, unsigned > > get_params();
  
  void check_map_keys(const std::map<char, std::map<std::string, unsigned> > & params);

  std::string get_string() const;
  

};  


HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic);

}
}

#endif


